models=(
    resnet18
    resnet34
    resnet50
    resnet101
    resnet152
);
for i in $models
do
	python memory_timeline.py --name $i --blocksize $1;
done
