"""vizmo entry point."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="vizmo - Real-time mesh-free data explorer")
    parser.add_argument("snapshot", help="Path to HDF5 snapshot file")
    parser.add_argument("--width", type=int, default=1920, help="Window width")
    parser.add_argument("--height", type=int, default=1080, help="Window height")
    parser.add_argument("--fov", type=float, default=90.0, help="Field of view in degrees")
    parser.add_argument(
        "--screenshot",
        type=str,
        default=None,
        metavar="OUT",
        help="Render one frame to OUT (PNG) after GPU init " "+ auto-range complete, then exit",
    )
    parser.add_argument("--fullscreen", action="store_true", help="Run in fullscreen mode at specified resolution")
    parser.add_argument("--no-stars", action="store_true", help="Disable star particle rendering")
    parser.add_argument(
        "--types",
        type=str,
        default=None,
        metavar="0,1,4",
        help="Comma-separated PartType numbers to load initially (default: gas only)",
    )
    parser.add_argument(
        "--center",
        type=str,
        default=None,
        metavar="X,Y,Z",
        help="Start the camera aimed at this point (snapshot coordinates)",
    )
    parser.add_argument(
        "--center-on",
        type=str,
        default=None,
        choices=["densest", "potential", "com", "median"],
        help="Auto-find the starting view center: highest-density particle, "
        "potential minimum, center of mass, or mass-weighted median",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        metavar="R",
        help="Initial camera distance from the view center (snapshot units)",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default=None,
        metavar="NAME",
        help="Starting colormap (e.g. magma, viridis, inferno)",
    )
    parser.add_argument(
        "--screenshot-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory for P-key screenshots and frame recordings (default: cwd)",
    )
    parser.add_argument(
        "--field",
        type=str,
        default=None,
        metavar="NAME",
        help="Starting field: raw (Masses, Density, ...) or derived "
        "(Temperature, NumberDensity, RadialVelocity, ...). With "
        "--mode SurfaceDensity it is the weight; with WeightedAverage/"
        "WeightedVariance it is the data field (mass-weighted).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["SurfaceDensity", "WeightedAverage", "WeightedVariance"],
        help="Starting render mode (default SurfaceDensity; "
        "WeightedAverage is implied when --field is a non-mass field)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        metavar="OUT",
        help="Profile the whole run with cProfile and dump " "stats to OUT (.pstats). View with snakeviz.",
    )
    args = parser.parse_args()

    types = None
    if args.types:
        types = [int(t) for t in args.types.split(",") if t.strip() != ""]
    center = None
    if args.center:
        parts = [float(c) for c in args.center.split(",")]
        if len(parts) != 3:
            parser.error("--center needs exactly three comma-separated values")
        center = parts

    from .wgpu_app import run_wgpu_app

    if args.profile:
        import cProfile
        import pstats

        pr = cProfile.Profile()
        pr.enable()
        try:
            run_wgpu_app(
                args.snapshot,
                width=args.width,
                height=args.height,
                fov=args.fov,
                fullscreen=args.fullscreen,
                screenshot=args.screenshot,
                no_stars=args.no_stars,
                types=types,
                center=center,
                center_on=args.center_on,
                radius=args.radius,
                colormap=args.colormap,
                screenshot_dir=args.screenshot_dir,
                field=args.field,
                mode=args.mode,
            )
        finally:
            pr.disable()
            pr.dump_stats(args.profile)
            stats = pstats.Stats(pr).sort_stats("cumulative")
            print("\n=== top 40 by cumulative time ===")
            stats.print_stats(40)
            print(f"\nFull profile written to {args.profile}")
            print(f"View with: snakeviz {args.profile}")
    else:
        run_wgpu_app(
            args.snapshot,
            width=args.width,
            height=args.height,
            fov=args.fov,
            fullscreen=args.fullscreen,
            screenshot=args.screenshot,
            no_stars=args.no_stars,
            types=types,
            center=center,
            center_on=args.center_on,
            radius=args.radius,
            colormap=args.colormap,
            screenshot_dir=args.screenshot_dir,
            field=args.field,
            mode=args.mode,
        )


if __name__ == "__main__":
    main()
