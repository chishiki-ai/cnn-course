if [ ! -f ~/.bashrc.bak ]
then
    cp ~/.bashrc ~/.bashrc.bak
else
    echo "bashrc.back exists, make sure you want to overwrite it. If you do, run < rm ~/.bashrc.bak > Exit"
    exit 1
fi

module load gcc/9.1.0 python3/3.9.2 
module save default
echo "source /scratch1/07980/sli4/training/cnn_course/bin/activate" > ~/.bashrc