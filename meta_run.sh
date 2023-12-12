target=50
window_size=1
for idx in 0
do
    cd secoexpan/
    ./run.sh ${target}
    cd ../contrastive
    ./run.sh ${window_size}
    cd ../
done
