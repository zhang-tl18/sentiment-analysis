# conda activate ai

python main.py --model MLP --train --epochs 40 --patience 7
python main.py --model CNN --train --epochs 40 --patience 7
python main.py --model RNN --train --epochs 40 --patience 7

python main.py --report