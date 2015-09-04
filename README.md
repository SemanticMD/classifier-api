# classifier-api
Self-contained web server for SemanticMD's classifiers.

## Requirements

The demo server requires Python with some dependencies.
To make sure you have the dependencies, please run `pip install -r examples/web_demo/requirements.txt`, and also make sure that you've compiled the Python Caffe interface and that it is on your `PYTHONPATH`.

Make sure that you have obtained the Reference Model and Auxiliary Data:

    ./scripts/download_model.py

## Run

Running `python examples/web_demo/app.py` will bring up the demo server, accessible at `http://0.0.0.0:5000`.
You can enable debug mode of the web server, or switch to a different port:

    % python examples/web_demo/app.py -h
    Usage: app.py [options]

    Options:
      -h, --help            show this help message and exit
      -d, --debug           enable debug mode
      -p PORT, --port=PORT  which port to serve content on

## Credits

Based on https://github.com/BVLC/caffe/tree/master/examples/web_demo.