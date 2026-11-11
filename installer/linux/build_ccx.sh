#!/usr/bin/env bash
# Sestaví CalculiX 2.23 pro Linux x86_64 ze zdrojů dhondt.de (bez práv správce, bez systémových balíků):
# překladač gcc / gfortran a ARPACK z conda-forge (micromamba), SPOOLES 2.2 z netlib, PARDISO přes Intel MKL.
#
#   installer/linux/build_ccx.sh VÝSTUP [PRACOVNÍ_SLOŽKA] [ARCHIV.tar.gz]
#
# VÝSTUP = složka, do které vzniknou: ccx_spooles (SPOOLES), ccx_i4 (SPOOLES + PARDISO – potřebuje libmkl_rt.so.2
# z pip balíčku mkl==2023.2.0, hledá se přes LD_LIBRARY_PATH, které nastaví program) a lib/ (libgomp, libquadmath
# – RUNPATH $ORIGIN/lib, takže binárky běží bez instalace čehokoli v systému; numerika je přilinkovaná staticky).
# ARCHIV (volitelně) = předsestavený balík pro distribuci (installer/linux/prebuilt/ccx_2.23_linux_x86_64.tar.gz) –
# jen ccx_spooles a lib/ (ccx_i4 s MKL se nešíří, GPL-2.0):
# vlastník 0:0 bez jmen, pevný čas (SOURCE_DATE_EPOCH), setříděné názvy, gzip bez názvu a času.
# Cesta pracovní složky (bývá v domovské složce → jméno uživatele) se do binárek nedostane: gcc ji v __FILE__
# a ladicích informacích mapuje -ffile-prefix-map na /build; gfortran zapisuje název zdrojového souboru do příkazů I/O
# (hlášení chyb za běhu) a -ffile-prefix-map tam NEPLATÍ (ověřeno gfortran 16.2) → LAPACK/ARPACK (cmake předává
# absolutní cesty) se překládají přes obal fc_rel, který zdroj z pracovní složky předá relativní cestou.
# Trvá několik minut; stahuje ~100 MB (zdrojáky + překladače).
set -euo pipefail

OUT="${1:?použití: build_ccx.sh VÝSTUP [PRACOVNÍ_SLOŽKA] [ARCHIV.tar.gz]}"
WORK="${2:-$(mktemp -d /tmp/fea_ccx_build.XXXXXX)}"   # bez pevné cesty v /tmp (podstrčení jiným uživatelem)
ARCHIVE="${3:-}"
CCX_VER=2.23
mkdir -p "$OUT" "$WORK"
OUT="$(cd "$OUT" && pwd)"
WORK="$(cd "$WORK" && pwd)"
if [ -n "$ARCHIVE" ]; then
    mkdir -p "$(dirname "$ARCHIVE")"
    ARCHIVE="$(cd "$(dirname "$ARCHIVE")" && pwd)/$(basename "$ARCHIVE")"
fi
cd "$WORK"
# pracovní složka (často v domovské složce uživatele) → neutrální /build ve všech řetězcích binárek
PMAP="-ffile-prefix-map=$WORK=/build"

get_mirrors() {  # get_mirrors SOUBOR SHA256 URL [ZÁLOŽNÍ_URL…] – první zdroj, který projde kontrolou otisku
    local file="$1" sum="$2" url; shift 2
    if [ -s "$file" ] && echo "$sum  $file" | sha256sum -c --status -; then return 0; fi   # i dřív stažené ověřit
    for url in "$@"; do
        echo "stahuji $url"
        if curl -fsSL --retry 3 -o "$file.part" "$url" && echo "$sum  $file.part" | sha256sum -c --status -; then
            mv "$file.part" "$file"; return 0
        fi
        echo "  nepodařilo se (nebo nesedí otisk) – zkouším další zdroj"
    done
    rm -f "$file.part"; echo "CHYBA: $file se nepodařilo stáhnout"; exit 1
}

# otisky SHA-256 ověřené 17. 10. 2026 (nová verze zdrojů = nový otisk, jinak sestavení skončí chybou)
get_mirrors ccx.src.tar.bz2 9c88385c10fb04f5dc6c4e98027a51bebdd8aee3920e05190d6c1dd08357d6e7 \
    "https://www.dhondt.de/ccx_${CCX_VER}.src.tar.bz2"
get_mirrors spooles.corr.tar.bz2 15528f09312dc7605c0600358f5e6de12945449f249dfcfca7417eed6c220b93 \
    "https://www.dhondt.de/ccx_${CCX_VER}.SPOOLEScorrection.tar.bz2"
# SPOOLES 2.2 z netlib; některé filtrující DNS (např. DNS4EU) netlib blokují → stejný soubor jako orig.tar.gz Debianu
get_mirrors spooles.tgz a84559a0e987a1e423055ef4fdf3035d55b65bbe4bf915efaa1a35bef7f8c5dd \
    "https://www.netlib.org/linalg/spooles/spooles.2.2.tgz" \
    "https://deb.debian.org/debian/pool/main/s/spooles/spooles_2.2.orig.tar.gz"
get_mirrors lapack.tar.gz eac9570f8e0ad6f30ce4b963f4f033f0f643e7c3912fc9ee6cd99120675ad48b \
    "https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.0.tar.gz"
get_mirrors arpack.tar.gz f6641deb07fa69165b7815de9008af3ea47eb39b2bb97521fbf74c97aba6e844 \
    "https://github.com/opencollab/arpack-ng/archive/refs/tags/3.9.1.tar.gz"
if [ ! -x mamba/bin/micromamba ]; then
    mkdir -p mamba
    # stejná připnutá verze a otisk jako MAMBA_URL / MAMBA_SHA256 v installer/fea_setup.py
    get_mirrors micromamba.tar.bz2 8761c382127e6363bd9e0a2451aa3ef90d071a79133f736e2f759a3bf13040dd \
        "https://github.com/mamba-org/micromamba-releases/releases/download/2.9.0-0/micromamba-linux-64.tar.bz2"
    tar -xjf micromamba.tar.bz2 -C mamba bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$WORK/mamba/root"
# sysroot glibc 2.28: bez něj conda-forge (gcc 16) přibalí sysroot 2.39 a binárky chtějí glibc ≥ 2.38
# (__isoc23_strtol/sscanf, fmod@GLIBC_2.38) – program má běžet od glibc 2.35 (Ubuntu 22.04, Debian 12 má 2.36)
TC_SPEC="gcc_linux-64 gfortran_linux-64 sysroot_linux-64=2.28 make cmake patchelf"
if [ ! -x "$WORK/tc/bin/cmake" ] || [ "$(cat "$WORK/tc/.fea_spec" 2>/dev/null)" != "$TC_SPEC" ]; then
    rm -rf "$WORK/tc"
    # shellcheck disable=SC2086
    mamba/bin/micromamba create -y -q -p "$WORK/tc" -c conda-forge $TC_SPEC
    printf '%s' "$TC_SPEC" > "$WORK/tc/.fea_spec"
fi
TC="$WORK/tc"
CC="$TC/bin/x86_64-conda-linux-gnu-gcc"
FC="$TC/bin/x86_64-conda-linux-gnu-gfortran"
AR="$TC/bin/x86_64-conda-linux-gnu-ar"
RANLIB="$TC/bin/x86_64-conda-linux-gnu-ranlib"
# ladicí informace staticky přilinkovaného libgfortran / libgcc a knihoven z conda-forge nesou cesty jejich sestavovací
# farmy (/home/conda/feedstock_root/…) – pryč; strip PŘED patchelf (strip po patchelf umí ELF poškodit)
STRIP="$TC/bin/x86_64-conda-linux-gnu-strip"
MAKE="$TC/bin/make"
# gcc 14+ povyšuje staré C konstrukce (SPOOLES, CalculiX) na chyby
PERMISSIVE="-Wno-error=int-conversion -Wno-error=incompatible-pointer-types -Wno-error=implicit-function-declaration -Wno-error=implicit-int -Wno-error=return-mismatch"

# ---- statické BLAS / LAPACK (referenční) a ARPACK → binárka nepotřebuje žádné sdílené numerické knihovny
export PATH="$TC/bin:$PATH"
FFL="-O2 -fPIC -fallow-argument-mismatch $PMAP"
# obal gfortranu pro cmake: zdrojový soubor z pracovní složky relativně k aktuální složce překladu (viz hlavička)
cat > "$WORK/fc_rel" <<EOF
#!/usr/bin/env bash
args=()
for a in "\$@"; do
    case "\$a" in
        "$WORK"/*.f|"$WORK"/*.F|"$WORK"/*.f90|"$WORK"/*.F90) [ -f "\$a" ] && a="\$(realpath --relative-to=. "\$a")" ;;
    esac
    args+=("\$a")
done
exec "$FC" "\${args[@]}"
EOF
chmod +x "$WORK/fc_rel"
STAMP="$FFL; fc_rel; $TC_SPEC"
# knihovny z dřívějšího běhu jen se stejným postupem (starší sestavení neslo cestu pracovní složky)
if [ ! -f static/lib/libarpack.a ] || [ "$(cat static/.flags 2>/dev/null)" != "$STAMP" ]; then
    rm -rf lapack-src arpack-src static && mkdir -p lapack-src arpack-src static
    tar -xzf lapack.tar.gz -C lapack-src --strip-components=1
    tar -xzf arpack.tar.gz -C arpack-src --strip-components=1
    cmake -S lapack-src -B lapack-src/b -G "Unix Makefiles" -DCMAKE_Fortran_COMPILER="$WORK/fc_rel" -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_MAKE_PROGRAM="$MAKE" -DCMAKE_AR="$AR" -DCMAKE_RANLIB="$RANLIB" -DCMAKE_INSTALL_PREFIX="$WORK/static" -DBUILD_SHARED_LIBS=OFF -DCBLAS=OFF \
        -DLAPACKE=OFF -DBUILD_TESTING=OFF -DCMAKE_Fortran_FLAGS="$FFL" -DCMAKE_C_FLAGS="-O2 -fPIC $PMAP" \
        -DCMAKE_INSTALL_LIBDIR=lib > "$WORK/lapack_cmake.log" 2>&1
    cmake --build lapack-src/b -j"$(nproc)" >> "$WORK/lapack_cmake.log" 2>&1
    cmake --install lapack-src/b >> "$WORK/lapack_cmake.log" 2>&1
    cmake -S arpack-src -B arpack-src/b -G "Unix Makefiles" -DCMAKE_Fortran_COMPILER="$WORK/fc_rel" -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_MAKE_PROGRAM="$MAKE" -DCMAKE_AR="$AR" -DCMAKE_RANLIB="$RANLIB" -DCMAKE_INSTALL_PREFIX="$WORK/static" -DBUILD_SHARED_LIBS=OFF -DMPI=OFF -DICB=OFF \
        -DEXAMPLES=OFF -DTESTS=OFF -DCMAKE_Fortran_FLAGS="$FFL" -DCMAKE_C_FLAGS="-O2 -fPIC $PMAP" -DCMAKE_INSTALL_LIBDIR=lib \
        -DBLAS_LIBRARIES="$WORK/static/lib/libblas.a" -DLAPACK_LIBRARIES="$WORK/static/lib/liblapack.a;$WORK/static/lib/libblas.a" \
        > "$WORK/arpack_cmake.log" 2>&1
    cmake --build arpack-src/b -j"$(nproc)" >> "$WORK/arpack_cmake.log" 2>&1
    cmake --install arpack-src/b >> "$WORK/arpack_cmake.log" 2>&1
    [ -f static/lib/libarpack.a ] || { tail -30 "$WORK/arpack_cmake.log"; exit 1; }
    printf '%s' "$STAMP" > static/.flags
fi
STATIC="$WORK/static/lib"

rm -rf src && mkdir src && cd src
tar -xjf ../ccx.src.tar.bz2                                  # CalculiX/ccx_2.23/src
mkdir SPOOLES.2.2 && tar -xzf ../spooles.tgz -C SPOOLES.2.2
tar -xjf ../spooles.corr.tar.bz2                              # oprava util.c podle dhondt.de
cp CalculiX/ccx_2.23/SPOOLES.2.2/I2Ohash/src/util.c SPOOLES.2.2/I2Ohash/src/util.c

# ---- SPOOLES: gcc, oprava makeGlobalLib (drawTree.c neexistuje), bez MPI
cd SPOOLES.2.2
sed -i "s|^  CC = .*|  CC = $CC -w -fcommon -O2 $PERMISSIVE $PMAP|" Make.inc
sed -i 's|drawTree.c|draw.c|g' Tree/src/makeGlobalLib
$MAKE -s lib 2>&1 | tail -3
[ -f spooles.a ] || { echo "SPOOLES se nepodařilo přeložit"; exit 1; }
cd ..

# ---- CalculiX
cd CalculiX/ccx_$CCX_VER/src
SP="$WORK/src/SPOOLES.2.2"
# Makefile dhondt.de má pevné LIBS s libarpack_INTEL.a a `ar vr`; použijeme vlastní Makefile.fea
cat > Makefile.fea <<EOF
include Makefile.inc
SCCXMAIN = ccx_${CCX_VER}.c
OCCXF = \$(SCCXF:.f=.o)
OCCXC = \$(SCCXC:.c=.o)
OCCXMAIN = \$(SCCXMAIN:.c=.o)
.c.o :
	\$(CC) \$(CFLAGS) -c \$<
.f.o :
	\$(FC) \$(FFLAGS) -c \$<
ccx_${CCX_VER}: \$(OCCXMAIN) ccx_${CCX_VER}.a
	\$(CC) \$(CFLAGS) -c ccx_${CCX_VER}.c; \$(FC) -O2 -o \$@ \$(OCCXMAIN) ccx_${CCX_VER}.a \$(LIBS) -fopenmp -static-libgfortran -static-libgcc
ccx_${CCX_VER}.a: \$(OCCXF) \$(OCCXC)
	\$(AR) rc \$@ \$?
EOF
MAKEFILE="-f Makefile.fea"
build_one() {
    local name="$1" cflags="$2" libs="$3"
    rm -f *.o ccx_${CCX_VER}.a ccx_${CCX_VER}
    $MAKE $MAKEFILE -j"$(nproc)" CC="$CC" FC="$FC" AR="$AR" \
        CFLAGS="-w -O2 -I$SP -DARCH=\"Linux\" -DSPOOLES -DARPACK -DMATRIXSTORAGE -DNETWORKOUT -fcommon $PERMISSIVE $PMAP $cflags" \
        FFLAGS="-w -O2 -cpp -fallow-argument-mismatch $PMAP" \
        LIBS="$SP/spooles.a $STATIC/libarpack.a $STATIC/liblapack.a $STATIC/libblas.a -lpthread -lm -lc $libs" \
        ccx_${CCX_VER} > "$WORK/build_$name.log" 2>&1 || { tail -30 "$WORK/build_$name.log"; exit 1; }
    cp ccx_${CCX_VER} "$OUT/$name"
    "$STRIP" --strip-debug "$OUT/$name"
    echo "hotovo: $OUT/$name"
}
build_one ccx_spooles "" ""
# PARDISO: -DPARDISO + Intel MKL (volání pardiso_ / pardisoinit_ řeší libmkl_rt.so.2 za běhu)
MKLDIR="${MKL_LIB_DIR:-}"
if [ -n "$MKLDIR" ] && [ -e "$MKLDIR/libmkl_rt.so.2" ]; then
    # pip balíček mkl nemá hlavičky; pardiso.c z nich potřebuje jen mkl_domain_set_num_threads a MKL_DOMAIN_PARDISO
    mkdir -p "$WORK/mklstub"
    cat > "$WORK/mklstub/mkl_service.h" <<'HDR'
#ifndef FEA_MKL_SERVICE_STUB
#define FEA_MKL_SERVICE_STUB
#define MKL_DOMAIN_PARDISO 4
int MKL_Domain_Set_Num_Threads(int nt, int mkl_domain);
#define mkl_domain_set_num_threads MKL_Domain_Set_Num_Threads   /* C rozhraní (mkl_service.h), ne Fortran _ */
#endif
HDR
    build_one ccx_i4 "-DPARDISO -I$WORK/mklstub" "-L$MKLDIR -l:libmkl_rt.so.2"
else
    echo "MKL_LIB_DIR (složka s libmkl_rt.so.2) není zadána – ccx_i4 (PARDISO) se nesestaví"
fi

# ---- sdílené knihovny z conda prostředí vedle binárek (RUNPATH $ORIGIN/lib), dokud přibývají závislosti
mkdir -p "$OUT/lib"
copy_deps() {  # copy_deps SOUBOR – zkopíruje závislosti ležící v $TC/lib
    LD_LIBRARY_PATH="$TC/lib" ldd "$1" | awk '/=> \//{print $3}' | { grep -F "$TC/" || true; } | while read -r so; do
        [ -e "$OUT/lib/$(basename "$so")" ] && continue
        cp -L "$so" "$OUT/lib/"
        "$STRIP" --strip-debug "$OUT/lib/$(basename "$so")"
    done
}
for b in "$OUT"/ccx_*; do
    [ -f "$b" ] || continue
    "$TC/bin/patchelf" --set-rpath '$ORIGIN/lib' "$b"
    copy_deps "$b"
done
for pass in 1 2 3; do
    for so in "$OUT"/lib/*.so*; do copy_deps "$so"; done
done
for so in "$OUT"/lib/*.so*; do "$TC/bin/patchelf" --set-rpath '$ORIGIN' "$so" 2>/dev/null || true; done
echo "knihovny:"; du -sh "$OUT/lib"; ls "$OUT" "$OUT/lib"

# ---- kontrola: cesta pracovní složky nemá v binárkách zůstat (např. starší překladač bez -ffile-prefix-map);
# u distribučního archivu je to chyba, u sestavení pro vlastní instalaci jen upozornění
for f in "$OUT"/ccx_* "$OUT"/lib/*; do
    [ -f "$f" ] || continue
    if grep -qaF "$WORK" "$f"; then
        echo "UPOZORNĚNÍ: $(basename "$f") obsahuje cestu pracovní složky sestavení"
        [ -z "$ARCHIVE" ] || exit 1
    fi
done
# ---- kontrola: nejvyšší požadovaná verze glibc (program podporuje od 2.35)
GLIBC_MAX=2.35
for b in "$OUT"/ccx_*; do
    [ -f "$b" ] || continue
    need=$("$TC/bin/x86_64-conda-linux-gnu-objdump" -T "$b" | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -uV | tail -1)
    echo "$(basename "$b"): vyžaduje glibc ≥ $need"
    if [ "$(printf '%s\n%s\n' "$need" "$GLIBC_MAX" | sort -V | tail -1)" != "$GLIBC_MAX" ]; then
        echo "UPOZORNĚNÍ: $(basename "$b") vyžaduje glibc $need > $GLIBC_MAX"
        [ -z "$ARCHIVE" ] || exit 1
    fi
done

# ---- distribuční archiv: bez uživatele a skupiny sestavujícího, pevný čas, stejné pořadí → opakovatelný obsah
if [ -n "$ARCHIVE" ]; then
    # bez ccx_i4: CalculiX (GPL-2.0) s proprietární Intel MKL se nesmí šířit – sestaví si ho instalátor u uživatele
    members=(); for f in ccx_spooles lib; do [ -e "$OUT/$f" ] && members+=("$f"); done
    tar -C "$OUT" --format=gnu --sort=name --owner=0 --group=0 --numeric-owner \
        --mtime="@${SOURCE_DATE_EPOCH:-1792022400}" --mode='u+rwX,go+rX,go-w' -cf - "${members[@]}" | gzip -9n > "$ARCHIVE"
    echo "archiv: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
fi
