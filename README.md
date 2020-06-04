open music sheet box bounding v1.0
======================
The goal of this project's to find bounding boxes into a music sheet.<br />
You'll have the possibility to either implement my API in your code and receive an array of bounding box or write them into a file.

## Easy dependencies installer

this installation script has been made on ubuntu 18.04.
Its use is to easily install python opencv, numpy....

```
bash build.bash
```

## Usage

The script can be found in src/

#### Example:
```
python3 src/sheet_to_bounding_box.py data/images/page0.jpg [nb threads]
```

## Features

- [x] .images
- [x] .pdfs
- [x] .threading
- [x] .return bounding boxes
- [ ] .write bouding boxes into file


## Examples 
### 1 thread

![](https://github.com/Cjdcoy/openmsbb/blob/master/data/demo.gif)

### 4 threads

![](https://github.com/Cjdcoy/openmsbb/blob/master/data/demo2.gif)

