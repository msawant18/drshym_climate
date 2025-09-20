# Dataset Card — DrShym Flood Tiles

**Sources**: Sentinel‑1 SAR IW GRD (VV). Labels from public flood polygons for selected event; weak labels derived via water occurrence + DEM masks.

**Preprocessing**: Speckle (Lee/median), normalization (z‑score), tiles 512 with 64 overlap.

**Splits**: train/val/test per `configs/flood.yaml`.

**Known issues**: DEM voids, inaccurate polygons near riverbanks, SAR artifacts.
