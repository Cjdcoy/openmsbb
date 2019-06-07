open music sheet box bounding
======================
The goald of this project's to find bounding boxes into a music sheet.<br />
You'll have the possibility to either implement my API in your code and receive an array of bounding box or write them into a file.

## Easy dependencies installer

this installation script has been made on ubuntu 18.04.
Its use is to easily install python opencv, numpy....

```
bash build.bash
```

## Usage

The script cap be found in src/

### 1 - run solution1
Each solution has many configuration parameters that you can see using ```-h``` flag
#### Examples:
Apply optic flow in realtime from webcam flux, previsualize and save the flux.<br /> 
```
cd src
python3 src/sheet_to_bounding_box.py data/images/page0.jpg
```


## Features

- [x] .images
- [x] .pdfs
- [x] .threading
- [ ] .return bounding boxes
- [ ] .write bouding boxes into file


![](https://github.com/Cjdcoy/openmsbb/data/demo.gif)
