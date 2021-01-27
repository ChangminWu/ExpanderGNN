# MIT License
# Copyright (c) 2020 Vijay Prakash Dwivedi, Chaitanya K. Joshi, Thomas Laurent, Yoshua Bengio, Xavier Bresson


# Command to download dataset:
#   bash download_datasets.sh


############
# ZINC
############

DIR=molecules/
cd $DIR

FILE=ZINC.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://data.dgl.ai/dataset/benchmarking-gnns/ZINC.pkl -o ZINC.pkl -J -L -k
fi

cd ..


FILE=ZINC-full.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://data.dgl.ai/dataset/benchmarking-gnns/ZINC-full.pkl -o ZINC.pkl -J -L -k
fi

cd ..


############
# MNIST and CIFAR10
############

DIR=superpixels/
cd $DIR


FILE=MNIST.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/wcfmo4yvnylceaz/MNIST.pkl?dl=1 -o MNIST.pkl -J -L -k
fi


FILE=CIFAR10.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/agocm8pxg5u8yb5/CIFAR10.pkl?dl=1 -o CIFAR10.pkl -J -L -k
fi