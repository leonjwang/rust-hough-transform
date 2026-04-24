Rust hough transform + floodfill single stroke detection

To run:

``
cargo run -- <input file> <output filepath> 1 1 <thickness>
``

Make ``<thickness>`` larger to detect less lines, smaller for more


ex: Run on black circle
``
cargo run -- data/circle.png data/ 1 1 520
``
