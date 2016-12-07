## File 
- Capstone.apk : Android application build result.
- Capstone.ipynb : Preprocessing SVHN dataset & develop convolutional network.
- Capstone.pdf : Capstone report.

## Preprocessing for create svhn_format_1.hdf5 file
- download & convert fuel's data
    - On Mac OS X/Windows : converting error => ValueError: Unable to dereference object (Can't insert duplicate key)
    - On mint linux 18 : converting success.


```
from fuel.downloaders.svhn import svhn_downloader
svhn_downloader(which_format=1, directory='./svhn/raw')
from fuel.converters.svhn import convert_svhn
convert_svhn(which_format=1, directory='./svhn/raw', output_directory='./svhn/converted')
```

