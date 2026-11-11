"""Tools: geometry, robust face selectors and named selections."""
from __future__ import annotations

from typing import Any

import numpy as np

from .common import (out, find_faces, face_summary, face_id, parse_face_id, resolve_faces, selection_info,
                     selection_users, selector_of)
from .session import FeaError, Session

MAX_FACES_OUT = 60


def part_info(sess: Session, pi: int) -> dict:
    from ..geometry import bbox
    p = sess.parts[pi]
    bb = bbox(p.shape)
    st = sess.study
    try:
        mat = st.material_of(p.name).name if st else None
    except KeyError:
        mat = None
    types: dict[str, int] = {}
    for fp in sess.face_props(pi):
        types[fp["type"]] = types.get(fp["type"], 0) + 1
    return {"name": p.name, "volume_mm3": p.volume(), "bbox_mm": {"min": bb[:3], "max": bb[3:]},
            "size_mm": bb[3:] - bb[:3], "faces": len(sess.face_props(pi)), "face_types": types,
            "material": mat}


def _neighbors(sess: Session, pi: int, fi: int) -> list[str]:
    """Adjacent faces (sharing an edge) – for the navigation of the LLM through the model."""
    try:
        from OCP.TopAbs import TopAbs_EDGE
        from OCP.TopExp import TopExp_Explorer

        def edges(f):
            exp, out_e = TopExp_Explorer(f, TopAbs_EDGE), []
            while exp.More():
                out_e.append(exp.Current())
                exp.Next()
            return out_e
        faces = sess.faces_of(pi)
        by_hash: dict[int, list] = {}            # shape hash = HashCode of OCCT (TShape + location, without orientation)
        for j, f in enumerate(faces):
            if j != fi:
                for e in edges(f):
                    by_hash.setdefault(hash(e), []).append((j, e))
        out_ids = {j for e in edges(faces[fi]) for j, g in by_hash.get(hash(e), []) if e.IsSame(g)}
        return [face_id(sess.parts[pi].name, j) for j in sorted(out_ids)]
    except Exception:  # noqa: BLE001 – only supplementary information
        return []


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_geometry_list_parts() -> str:
        """Parts (bodies): name, volume [mm3], bounding box [mm], face counts/types, material."""
        with sess.use():
            sess.require_study()
            return out({"parts": [part_info(sess, i) for i in range(len(sess.parts))]})

    @mcp.tool(structured_output=False)
    def fea_geometry_find_faces(part: str | None = None, x: float | None = None, y: float | None = None,
                                z: float | None = None, normal: list[float] | None = None,
                                plane: dict | None = None, cylinder: dict | None = None,
                                radius: float | None = None, where: str | None = None,
                                type: str | None = None, near: list[float] | None = None,
                                limit: int | None = None, tol: float = 1e-6) -> str:
        """Finds faces by geometry; returns stable selectors that survive re-meshing. Conditions are ANDed: part; x/y/z (face lies in plane x=..., by envelope, mm; tol relative); normal [nx,ny,nz] (planar face, outward normal); plane {point,normal}; cylinder {point,axis,radius?} (coaxial cylindrical faces = whole hole/pin); radius; type plane|cylinder|cone|sphere|torus; where = expression over area, type, cx cy cz, nx ny nz, xmin..zmax, radius, e.g. "type == 'cylinder' and radius < 5 and cz > 10"; near [x,y,z] sorts by centroid distance; limit. Output ids 'Díl#i' usable directly as selections (single or list) in other tools."""
        with sess.use():
            sess.require_study()
            hits = find_faces(sess, part, type, x, y, z, normal, plane, cylinder, radius, where, near, limit, tol)
            faces = [face_summary(sess.parts[pi].name, fi, sess.face_props(pi)[fi]) for pi, fi in hits]
            d: dict[str, Any] = {"count": len(faces), "total_area_mm2": sum(f["area_mm2"] for f in faces),
                                 "faces": faces[:MAX_FACES_OUT]}
            if len(faces) > MAX_FACES_OUT:
                d["note"] = f"The first {MAX_FACES_OUT} shown – narrow the query (part, where, limit)."
            if not faces:
                d["hint"] = ("Nothing found. Check the dimensions of the part (fea_geometry_list_parts), the x/y/z coordinates must lie exactly on the face; for an approximate search use where or near.")
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_geometry_get_face_info(selector: str | dict) -> str:
        """Face details for an id 'Díl#i' or query: type, area, centroid, point+normal, envelope, cylinder axis/radius, neighbouring faces, selections containing it, full selector JSON."""
        with sess.use():
            st = sess.require_study()
            if isinstance(selector, str) and "#" in selector:
                pi, fi = parse_face_id(sess, selector)
            else:
                fs = resolve_faces(sess, selector)
                if len(fs) != 1:
                    raise FeaError(f"The selector matches {len(fs)} faces – give one")
                from ..selectors import resolve
                r = resolve(fs[0], sess.parts)
                if r is None:
                    raise FeaError("The selector face was not found in the geometry")
                pi, fi = r
            p = sess.face_props(pi)[fi]
            pname = sess.parts[pi].name
            d = face_summary(pname, fi, p, full=True)
            d["part"] = pname
            d["neighbors"] = _neighbors(sess, pi, fi)
            sel = selector_of(sess, pi, fi)
            d["in_selections"] = [k for k, s in st.selections.items()
                                  if any(f.part == pname and abs(f.area - sel.area) <= 1e-6 * max(sel.area, 1)
                                         and np.linalg.norm(np.subtract(f.centroid, sel.centroid)) < 1e-6 for f in s.faces)]
            d["selector"] = sel.to_dict()
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_selection_create(name: str, faces: str | list | dict) -> str:
        """Creates/overwrites a named face selection. faces = id 'Díl#i', list of ids, query as in fea_geometry_find_faces (e.g. {"x": 0}), selector, or another selection name."""
        with sess.use():
            st = sess.require_study()
            st.add_selection(name, resolve_faces(sess, faces))
            sess.changed()
            return out(selection_info(sess, name))

    @mcp.tool(structured_output=False)
    def fea_geometry_find_edges(part: str | None = None, x: float | None = None, y: float | None = None,
                                z: float | None = None, radius: float | None = None,
                                direction: list[float] | None = None, type: str | None = None,
                                where: str | None = None, near: list[float] | None = None,
                                limit: int | None = None, tol: float = 1e-6) -> str:
        """Finds edges (for fea_load_add_edge_force). ANDed: part; x/y/z (edge lies in plane; two coordinates = line); radius (circle/arc); direction [dx,dy,dz] (parallel segments); type line|circle|ellipse|curve; where over length, type, cx cy cz, xmin..zmax, radius; near; limit. Output: id 'Díl#e<i>', length, centroid, end points, circle centre/axis."""
        from .common import find_edges as _find_edges, edge_summary
        with sess.use():
            sess.require_study()
            hits = _find_edges(sess, part, type, x, y, z, radius, direction, near, limit, tol, where)
            edges = [edge_summary(sess.parts[pi].name, ei, sess.edge_props(pi)[ei]) for pi, ei in hits]
            d: dict[str, Any] = {"count": len(edges), "total_length_mm": sum(e["length_mm"] for e in edges),
                                 "edges": edges[:MAX_FACES_OUT]}
            if len(edges) > MAX_FACES_OUT:
                d["note"] = f"The first {MAX_FACES_OUT} shown – narrow the query (part, where, limit)."
            if not edges:
                d["hint"] = "Nothing found – the x/y/z coordinates must lie exactly on the edge; try near or where."
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_selection_create_edges(name: str, edges: str | list | dict) -> str:
        """Creates/overwrites a named EDGE selection. edges = id 'Díl#e<i>', list, query as in fea_geometry_find_edges (e.g. {"x": 200, "z": 10}) or another edge selection name."""
        from .common import resolve_edges
        with sess.use():
            st = sess.require_study()
            if name in st.selections and selection_users(st, name):
                raise FeaError(f"Selection '{name}' is in use ({selection_users(st, name)[0][0]}) – choose another name")
            st.add_selection(name, edges=resolve_edges(sess, edges))
            sess.changed()
            return out(selection_info(sess, name))

    @mcp.tool(structured_output=False)
    def fea_selection_create_points(name: str, points: list | dict, part: str | None = None) -> str:
        """Creates/overwrites a named selection of POINTS (point force). points = [x,y,z], list of points or {"point","part"}; each is projected onto the nearest surface of `part` (None = nearest part) and becomes a mesh node."""
        from .common import resolve_points
        with sess.use():
            st = sess.require_study()
            if name in st.selections and selection_users(st, name):
                raise FeaError(f"Selection '{name}' is in use ({selection_users(st, name)[0][0]}) – choose another name")
            pts, dists = resolve_points(sess, points, part)
            st.add_selection(name, points=pts)
            sess.changed()
            d = selection_info(sess, name)
            if max(dists, default=0.0) > 1e-6:
                d["projected_mm"] = dists
            return out(d)

    @mcp.tool(structured_output=False)
    def fea_selection_list() -> str:
        """Named face selections (faces, parts, centroid) and where they are used."""
        with sess.use():
            st = sess.require_study()
            return out({"selections": [{**selection_info(sess, k), "used_by": [w for w, _ in selection_users(st, k)]}
                                       for k in st.selections]})

    @mcp.tool(structured_output=False)
    def fea_selection_delete(name: str) -> str:
        """Deletes a selection (refused while used by a constraint, load, thermal condition, refinement, mass, probe, contact, beam connection or mechanism)."""
        with sess.use():
            st = sess.require_study()
            if name not in st.selections:
                raise FeaError(f"Selection '{name}' does not exist")
            users = selection_users(st, name)
            if users:
                raise FeaError(f"Selection '{name}' is in use ({users[0][0]}) – first remove the item that uses it")
            del st.selections[name]
            sess.changed()
            return out({"deleted": name})

    @mcp.tool(structured_output=False)
    def fea_imprint_add(name: str, part: str, shape: str = "circle",
                        center: list[float] | None = None, normal: list[float] | None = None,
                        radius: float = 10.0, width: float = 10.0, height: float = 10.0,
                        points: list[list[float]] | None = None, size_mm: float | None = None) -> str:
        """Imprint/split face: splits a planar part face into a new named face (also a selection) for loads, constraints, contacts. name, part, shape circle|rectangle|polygon, center [x,y,z] mm, normal (default [0,0,1]), radius [mm] (circle), width/height [mm] (rectangle, axes of the face frame x' = n x X, y' = n x x'; rotation not supported). polygon: points = [[u,v],...] in the face frame around center, or [[x,y,z],...] global points projected onto their plane (center/normal then optional). size_mm = element size on the imprint."""
        with sess.use(op="imprint_add"):
            st = sess.require_study()
            if any(imp.name == name for imp in st.imprints):
                raise FeaError(f"An imprint named '{name}' already exists")
            from ..imprint import Imprint, ImprintError
            try:
                if shape == "polygon" and points and all(len(p) == 3 for p in points):
                    imp = Imprint.polygon_from_3d(name, part, points, normal=normal, size=size_mm)
                else:
                    if points and any(len(p) != 2 for p in points):
                        raise ImprintError("points: all [u, v] (face frame) or all [x, y, z] (global)")
                    imp = Imprint(
                        name=name,
                        part=part,
                        shape=shape,
                        center=list(center) if center else [0.0, 0.0, 0.0],
                        normal=list(normal) if normal else [0.0, 0.0, 1.0],
                        radius=float(radius),
                        dx=float(width),
                        dy=float(height),
                        size=float(size_mm) if size_mm else None,
                        points=[[float(u), float(v)] for u, v in points] if shape == "polygon" and points else [],
                    )
                    imp.validate()
            except ImprintError as e:
                raise FeaError(str(e)) from None
            if imp.part not in sess.part_names():
                raise FeaError(f"Part '{imp.part}' does not exist. Parts: {', '.join(sess.part_names())}")
            st.imprints.append(imp)
            sess.changed(mesh=True)  # the imprint requires regenerating the mesh
            return out({"added_imprint": imp.to_dict(), "total_imprints": len(st.imprints)})

    @mcp.tool(structured_output=False)
    def fea_imprint_list() -> str:
        """Lists the imprints."""
        with sess.use():
            st = sess.require_study()
            return out({"imprints": [imp.to_dict() for imp in st.imprints]})

    @mcp.tool(structured_output=False)
    def fea_imprint_delete(name: str) -> str:
        """Deletes an imprint by name."""
        with sess.use(op="imprint_delete"):
            st = sess.require_study()
            idx = next((i for i, imp in enumerate(st.imprints) if imp.name == name), None)
            if idx is None:
                raise FeaError(f"Imprint '{name}' does not exist")
            deleted = st.imprints.pop(idx)
            sess.changed(mesh=True)
            return out({"deleted": deleted.to_dict(), "remaining": len(st.imprints)})

