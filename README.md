# CloudyGo

Frontend for displaying MiniGo Training data.

*This is not an official Google product.*

## Getting Started

Checkout the [CloudyGo.com](http://CloudyGo.com) to see the running site.

Local development requires mirroring large number of sgf files and setting up
local database. More on this later.

TODO(sethtroisi): Add general setup instructions.


### Prerequisites

The site requires several python libraries, this may not be a complete list
```
pip3 install choix, flask, numpy, tqdm
```

### coding style 

Style guide is a mix of Google Python + PEP8,
Some older code may not be perfectly compliant.

## Deployment

CloudyGo.com is run by Seth Troisi, local deployment is normally tested with
```
FLASK_DEBUG=1 FLASK_APP="web/serve.py" flask run --host 0.0.0.0 --port 6000
```

## Contributing

Please read [CONTRIBUTING](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

## Authors

See also the list of [AUTHORS](AUTHORS) who participated in this project.

## License

This project is licensed under the Apache 2 License - see [LICENSE](LICENSE) file for details

## Acknowledgments

* [MiniGo](https://github.com/tensorflow/minigo)
* Andrew Jackson for his infinite patience with my questions

