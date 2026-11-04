"""Command line interface.

  python cli.py demo  --out out_demo                       # synthetic demo, all scenarios
  python cli.py zones baseline.jpg --out zones.json        # draw zones with the mouse
  python cli.py check baseline.jpg new.jpg --zones zones.json --out out/
"""
import argparse
import json
import os

import cv2

from roomcheck import Params, check_room, load_zones, save_zones
from roomcheck import demo_data
from roomcheck.zones import select_zones_interactive


def _print_report(rep):
    a = rep.alignment
    print(f"  alignment: {'OK' if a.ok else 'FAILED - ' + a.reason}  "
          f"(kp {a.n_kp_base}/{a.n_kp_new}, good {a.n_good}, inliers {a.n_inliers})")
    for r in rep.results:
        print(f"  [{r.status:9s}] {r.name:22s} item={r.item:14s} score={r.score:5.2f} "
              f"changed={r.change_ratio:5.2f} lost={r.lost_edges:5.2f} gained={r.gained_edges:5.2f}")


def _save(rep, out):
    os.makedirs(out, exist_ok=True)
    for k, v in rep.images.items():
        if v is not None:
            cv2.imwrite(os.path.join(out, f"{k}.png"), v)
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(rep.to_dict(), f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo"); d.add_argument("--out", default="out_demo"); d.add_argument("--method", default="sift")
    z = sub.add_parser("zones"); z.add_argument("baseline"); z.add_argument("--out", default="zones.json")
    c = sub.add_parser("check"); c.add_argument("baseline"); c.add_argument("new")
    c.add_argument("--zones", required=True); c.add_argument("--out", default="out")
    c.add_argument("--method", default="sift", choices=["sift", "orb"])
    c.add_argument("--color-thresh", type=float, default=Params.color_thresh)
    c.add_argument("--zone-thresh", type=float, default=Params.zone_change_thresh)
    a = ap.parse_args()

    if a.cmd == "demo":
        zones = demo_data.default_zones()
        os.makedirs(a.out, exist_ok=True)
        save_zones(zones, os.path.join(a.out, "zones_demo.json"))
        for i, name in enumerate(demo_data.SCENARIOS):
            base, new = demo_data.make_pair(name)
            sd = os.path.join(a.out, f"scenario{i}")
            os.makedirs(sd, exist_ok=True)
            cv2.imwrite(os.path.join(sd, "baseline.png"), base)
            cv2.imwrite(os.path.join(sd, "new.png"), new)
            print(f"\n== {name}")
            rep = check_room(base, new, zones, Params(method=a.method))
            _print_report(rep)
            _save(rep, sd)
    elif a.cmd == "zones":
        img = cv2.imread(a.baseline)
        save_zones(select_zones_interactive(img), a.out)
        print("saved", a.out)
    else:
        base, new = cv2.imread(a.baseline), cv2.imread(a.new)
        if base is None or new is None:
            raise SystemExit("could not read one of the images")
        p = Params(method=a.method, color_thresh=a.color_thresh, zone_change_thresh=a.zone_thresh)
        rep = check_room(base, new, load_zones(a.zones), p)
        _print_report(rep)
        _save(rep, a.out)
        print("missing:", [r.item for r in rep.missing] or "nothing")


if __name__ == "__main__":
    main()
