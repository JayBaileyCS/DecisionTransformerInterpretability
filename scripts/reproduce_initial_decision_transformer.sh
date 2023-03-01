
# Decent length training run that gets performance close to the paper
python src/run_decision_transformer.py --exp_name Test --trajectory_path trajectories/MiniGrid-Dynamic-Obstacles-8x8-v0bd60729d-dc0b-4294-9110-8d5f672aa82c.pkl --d_model 128 --n_heads 2 --d_mlp 256 --n_layers 1 --learning_rate 0.0001 --batch_size 128 --train_epochs 5000 --test_epochs 10 --n_ctx 3 --pct_traj 1 --weight_decay 0.001 --seed 1 --wandb_project_name DecisionTransformerInterpretability --test_frequency 1000 --eval_frequency 1000 --eval_episodes 10 --initial_rtg 1 --prob_go_from_end 0.1 --eval_max_time_steps 1000 --track True

# Shorter Version for testing. Should still see successful agent.
python -m src.run_decision_transformer --exp_name Test --trajectory_path trajectories/MiniGrid-Dynamic-Obstacles-8x8-v0bd60729d-dc0b-4294-9110-8d5f672aa82c.pkl --d_model 128 --n_heads 2 --d_mlp 256 --n_layers 1 --learning_rate 0.0001 --batch_size 128 --train_epochs 500 --test_epochs 10 --n_ctx 3 --pct_traj 1 --weight_decay 0.001 --seed 1 --wandb_project_name DecisionTransformerInterpretability --test_frequency 100 --eval_frequency 100 --eval_episodes 10 --initial_rtg 1 --prob_go_from_end 0.1 --eval_max_time_steps 1000 --track True
