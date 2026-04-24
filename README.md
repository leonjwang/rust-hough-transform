Rust hough transform + floodfill single stroke detection

To run:

``
cargo run -- <input file> <output filepath> 1 1 <offset>
``

Make ``<offset>`` larger to detect more lines, smaller for less


ex: Run on bullseye (input.png)
``
cargo run -- data/input.png data/ 1 1 50
``
