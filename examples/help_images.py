"""Pictures of the example models for the help (`fea_gui/help/ukazka*.png` in English, `fea_gui/help/cs/` in Czech).

    python -m examples.help_images [--lang en cs] [--only 3 6]

The examples `examples/ukazka*.fea` keep their stored (Czech) names of parts, selections and load cases. For the English
pictures these names are translated through `NAMES` only in the picture – the projects are not changed. A text of the
English picture with a Czech letter is reported (add it to `NAMES`).
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELP = ROOT / "fea_gui" / "help"
SIZE = {"en": (1280, 720), "cs": (1100, 720)}   # English legends are longer

# example number → (project, picture, render call)
PICTURES = {
    1: ("example1_bracket.fea", "example1_model.png", "model"),
    2: ("example2_assembly.fea", "example2_model.png", "model"),
    3: ("example3_flange.fea", "example3_model.png", "model"),
    4: ("example4_jib.fea", "example4_model.png", "model"),
    5: ("example5_heat_sink.fea", "example5_model.png", "model"),
    6: ("example6_hall_frame.fea", "example6_model.png", "frame"),
    7: ("example7_pcb_dynamics.fea", "example7_model.png", "model"),
    8: ("example8_crank.fea", "example8_model.png", "model"),
    9: ("example9_base_contact.fea", "example9_model.png", "model"),
    10: ("example10_bracket_topology.fea", "example10_model.png", "model"),
    11: ("example10_bracket_topology.fea", "example10_density.png", "density"),
}

# stored Czech names of the examples → English (only for the pictures); longer names first when replacing
NAMES = {
    "Ukázka 1 – tištěný držák": "Example 1 – printed bracket",
    "Ukázka 2 – sestava ocel + tisk": "Example 2 – steel + printed assembly",
    "Ukázka 3 – přírubový spoj": "Example 3 – flange joint",
    "Ukázka 4 – svařovaný výložník": "Example 4 – welded jib",
    "Ukázka 5 – chladič elektroniky": "Example 5 – electronics heat sink",
    "Ukázka 6 – rám haly z prutů": "Example 6 – hall frame of members",
    "Ukázka 7 – dynamika přípravku s DPS": "Example 7 – dynamics of a fixture with a PCB",
    "Ukázka 8 – klikový mechanismus": "Example 8 – crank mechanism",
    "Ukázka 9 – šroubovaná patka, spára jako kontakt": "Example 9 – bolted base, joint as contact",
    "Ukázka 10 – konzola, topologická optimalizace": "Example 10 – bracket, topology optimization",
    "Boční síla 800 N": "Side force 800 N",
    "Boční síla na sloupek": "Side force on the post",
    "Břemeno 4 kN": "Load 4 kN",
    "Břemeno 5000 N": "Load 5000 N",
    "Celo vylozniku": "Jib end face",
    "DPS (FR-4)": "PCB (FR-4)",
    "Dolní vrut": "Lower screw",
    "Horní vrut": "Upper screw",
    "Hlava sloupku": "Post head",
    "Horni celo vika": "Cover top face",
    "Konce základové desky": "Base plate ends",
    "Kotva": "Anchor",
    "MBD max reakce": "MBD max reaction",
    "Ochlazovany povrch": "Cooled surface",
    "Ohřev za provozu": "Heating in operation",
    "Ojnice": "Connecting rod",
    "Oko ramena": "Arm eye",
    "Oko vylozniku": "Jib eye",
    "Opření o stěnu": "Support on the wall",
    "Osazení DPS": "PCB components",
    "Osový tah potrubí 10 kN": "Axial pipe tension 10 kN",
    "Otvor sloupku": "Post hole",
    "Otvor": "Hole",
    "Patice transformatoru": "Transformer base",
    "Patice transformátoru": "Transformer base",
    "Povrch ramena": "Arm surface",
    "Rameno (tisk PLA)": "Arm (PLA print)",
    "Rameno": "Arm",
    "Sloupek": "Post",
    "Sníh": "Snow",
    "Spodek modulu": "Module bottom",
    "Spodek patky": "Base bottom",
    "Spodek trubky": "Pipe bottom",
    "Síla na oko ramena": "Force on the arm eye",
    "Tah 20 kN": "Tension 20 kN",
    "Upnutí na stůl vibrátoru": "Clamping on the shaker table",
    "Vlastní tíha": "Self-weight",
    "Vnitrek trubky": "Pipe inside",
    "Vnitřní tlak 2,5 MPa": "Internal pressure 2.5 MPa",
    "Vykonovy modul": "Power module",
    "Vítr zleva": "Wind from the left",
    "Zadni stena desky": "Plate back wall",
    "Zadní plocha": "Back face",
    "Zatížení oka 200 N": "Eye load 200 N",
    "setrvačnost a tíha (MBD)": "inertia and weight (MBD)",
    "Čep ojnice": "Connecting rod pin",
    "Čep pístu": "Piston pin",
    "Šroub": "Bolt",
    "Oko": "Eye",
}
NAMES_EXTRA = {                           # thermal states, probes, masses, contacts, members, part names
    "Provoz – ustálený stav": "Operation – steady state",
    "Náběh – 1 hodina": "Start-up – 1 hour",
    "Náběh – 10 minut": "Start-up – 10 minutes",
    "Transformátor 80 g": "Transformer 80 g",
    "DPS vlevo od transformátoru": "PCB left of the transformer",
    "Horní hrana konzoly": "Bracket top edge",
    "Transformátor (patice)": "Transformer (base)",
    "Spára Patka sloupku–Zakladova deska": "Joint post base–base plate",
    "Patka sloupku": "Post base",
    "Zakladova deska": "Base plate",
    "příčel L": "crossbar L",
    "příčel P": "crossbar R",
    "ztužidlo": "brace",
    "sloup L": "column L",
    "sloup P": "column R",
    "vaznice": "purlin",
    "Chladic (Al)": "Heat sink (Al)",
    "Drzak (tisk PLA)": "Bracket (PLA print)",
    "Klika": "Crank",
    "Konzola (navrhovy prostor)": "Bracket (design space)",
    "Konzola (Al)": "Bracket (Al)",
    "Kotevni deska": "Anchor plate",
    "Pist": "Piston",
    "Priruba potrubi": "Pipe flange",
    "Rameno vylozniku": "Jib arm",
    "Svar a4 rameno horni": "Weld a4 arm top",
    "Svar a4 zebro svisly vlevo": "Weld a4 rib vertical left",
    "Svar a4 zebro svisly vpravo": "Weld a4 rib vertical right",
    "Svar a4 zebro vodorovny vlevo": "Weld a4 rib horizontal left",
    "Svar a4 zebro vodorovny vpravo": "Weld a4 rib horizontal right",
    "Svar a4 vlevo": "Weld a4 left",
    "Svar a4 vpravo": "Weld a4 right",
    "Viko priruby": "Flange cover",
    "Zebro vylozniku": "Jib rib",
    "Patka": "Base",
    "Ram": "Frame",
}

CZECH = re.compile(r"[áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]")


def _translate(text: str, table: list[tuple[str, str]]) -> str:
    """One pass over the text, longest name first, whole words only (a translated text is not translated again)."""
    pattern = re.compile("|".join(rf"(?<!\w){re.escape(a)}(?!\w)" for a, _ in table))
    lookup = dict(table)
    return pattern.sub(lambda m: lookup[m.group(0)], text)


def _patch_render(lang: str, leftovers: list[str]):
    """Translates the names in every text of the scene just before drawing (English only)."""
    import fea_core.mcp.render as rmod
    orig = rmod.render
    table = sorted({**NAMES, **NAMES_EXTRA}.items(), key=lambda kv: -len(kv[0]))

    def tr_text(s):
        if not isinstance(s, str):
            return s
        if lang == "en":
            s = _translate(s, table)
            if CZECH.search(s):
                leftovers.append(s)
        return s

    def render(scene, *a, **kw):
        scene.title = [tr_text(t) for t in scene.title]
        scene.info = [tr_text(t) for t in scene.info]
        scene.legend_title = tr_text(scene.legend_title)
        scene.swatches = [(c, tr_text(t), *rest) for c, t, *rest in scene.swatches]
        for group in (scene.markers, scene.arrows):
            for m in group:
                if "text" in m:
                    m["text"] = tr_text(m["text"])
        return orig(scene, *a, **kw)

    rmod.render = render
    return lambda: setattr(rmod, "render", orig)


def make(lang: str, only: list[int] | None = None) -> list[str]:
    from fea_core import i18n
    from fea_core.mcp.session import Session
    from fea_core.rest.registry import collect_tools

    out_dir = HELP if lang == "en" else HELP / "cs"
    leftovers: list[str] = []
    restore = _patch_render(lang, leftovers)
    try:
        with i18n.using(lang), tempfile.TemporaryDirectory() as tmp:
            if lang == "cs":
                i18n.patch_matplotlib()
            sess = Session(Path(tmp))
            tools = collect_tools(sess)

            def call(name, **kw):
                t = tools[name]
                return t.fn(**t.validate(kw))

            for k, (fea, png, kind) in PICTURES.items():
                if only and k not in only:
                    continue
                call("fea_study_open", path=str(ROOT / "examples" / fea))
                target = out_dir / png
                w, h = SIZE[lang]
                if kind == "model":
                    call("fea_model_render_image", output_path=str(target), view="multi", width=w, height=h,
                         return_image=False)
                elif kind == "frame":
                    call("fea_frame_render", quantity="model", output_path=str(target), view="iso", width=w,
                         height=h, show_names=False, return_image=False)
                else:
                    call("fea_results_render_image", output_path=str(target), field="topo_density",
                         deformed=False, width=w, height=h, return_image=False)
                print(f"  {lang}: {target.relative_to(ROOT)}")
    finally:
        restore()
    return leftovers


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lang", nargs="+", default=["en", "cs"], choices=["en", "cs"])
    ap.add_argument("--only", nargs="+", type=int, help="picture numbers (keys of PICTURES)")
    a = ap.parse_args(argv)
    bad = []
    for lang in a.lang:
        bad += make(lang, a.only)
    for s in sorted(set(bad)):
        print(f"  ! Czech text in an English picture: {s}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
