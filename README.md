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

![img](https://github.com/leonjwang/rust-hough-transform/blob/master/data/input.png)

![img](https://github.com/leonjwang/rust-hough-transform/blob/master/data/input.png-0.png)
![img](https://github.com/leonjwang/rust-hough-transform/blob/master/data/input.png-0-lines.png)
![img](https://github.com/leonjwang/rust-hough-transform/blob/master/data/input.png-1.png)
![img](https://github.com/leonjwang/rust-hough-transform/blob/master/data/input.png-1-lines.png)
![img](https://github.com/leonjwang/rust-hough-transform/blob/master/data/input.png-2.png)
![img](https://github.com/leonjwang/rust-hough-transform/blob/master/data/input.png-2-lines.png)

Does not like shapes which aren't outlines

