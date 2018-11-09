FROM andrewosh/binder-base
MAINTAINER Manuel Castejón-Limas <manuel.castejon@gmail.com>
USER root
RUN apt-get -y update && apt-get install -y graphviz
RUN conda install pandas numpy matplotlib scipy networkx pygraphviz
RUN pip install https://github.com/mcasl/misc/archive/master.zip
