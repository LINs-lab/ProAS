from pathlib import Path


GRAPH_GUIDE_DIR = Path(__file__).resolve().parents[1]


def test_fair_alfworld_migpo_graph_guide_v2_launcher_uses_native_milestone_credit():
    path = (
        GRAPH_GUIDE_DIR
        / "run_sh_fair"
        / "1_5B"
        / "seed0"
        / "alfworld"
        / "migpo_graph_guide_v2_milestone_future_miss_no_dis_mask_alfworld.sh"
    )
    script = path.read_text()

    assert '"$PYTHON_BIN" -m recipe.graph_guide.main_alfworld_graph_guide' in script
    assert "algorithm.adv_estimator=migpo" in script
    assert "algorithm.gamma=0.95" in script
    assert "algorithm.migpo.step_advantage_w=10.0" in script
    assert "algorithm.migpo.mode=mean_norm" in script
    assert "algorithm.migpo.gamma=0.9" in script
    assert "algorithm.migpo.threshold=0.95" in script
    assert "++algorithm.graph_guide.env_adapter=migpo_native" in script
    assert "++algorithm.graph_guide.annotation_source=migpo_milestone" in script
    assert "++algorithm.graph_guide.migpo_reasonable_source=next_hit_segment" in script
    assert "++algorithm.graph_guide.migpo_threshold=0.95" in script
    assert "++algorithm.graph_guide.disadvantage_mode=advantage_cluster" in script
    assert "++algorithm.graph_guide.blame_source=milestone_future_miss" in script
    assert "++algorithm.graph_guide.mask_positive_unreasonable=true" in script
    assert "++algorithm.graph_guide.mask_negative_reasonable=false" in script
    assert "++algorithm.graph_guide.negative_mask_mode=none" in script
    assert "trainer.test_freq=5" in script
    assert "trainer.save_freq=50" in script


def test_fair_7b_alfworld_migpo_graph_guide_v2_launcher_uses_native_milestone_credit():
    path = (
        GRAPH_GUIDE_DIR
        / "run_sh_fair"
        / "7B_new"
        / "seed0"
        / "alfworld"
        / "migpo_graph_guide_v2_milestone_future_miss_no_dis_mask_alfworld.sh"
    )
    script = path.read_text()

    assert '"$PYTHON_BIN" -m recipe.graph_guide.main_alfworld_graph_guide' in script
    assert "setup_verl_7b_env alfworld" in script
    assert 'model_path="${MODEL_PATH_7B}"' in script
    assert "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-4,5,6,7}" in script
    assert "algorithm.adv_estimator=migpo" in script
    assert "algorithm.gamma=0.95" in script
    assert "algorithm.migpo.step_advantage_w=10.0" in script
    assert "algorithm.migpo.mode=$mode" in script
    assert "algorithm.migpo.gamma=0.9" in script
    assert "algorithm.migpo.threshold=0.95" in script
    assert "++algorithm.graph_guide.env_adapter=migpo_native" in script
    assert "++algorithm.graph_guide.annotation_source=migpo_milestone" in script
    assert "++algorithm.graph_guide.migpo_reasonable_source=next_hit_segment" in script
    assert "++algorithm.graph_guide.migpo_threshold=0.95" in script
    assert "++algorithm.graph_guide.disadvantage_mode=advantage_cluster" in script
    assert "++algorithm.graph_guide.blame_source=milestone_future_miss" in script
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=4" in script
    assert "trainer.n_gpus_per_node=4" in script
    assert "trainer.test_freq=5" in script
    assert "trainer.save_freq=50" in script


def test_fair_weshop_migpo_graph_guide_v2_launcher_uses_native_milestone_credit():
    path = (
        GRAPH_GUIDE_DIR
        / "run_sh_fair"
        / "1_5B"
        / "seed0"
        / "weshop"
        / "migpo_graph_guide_v2_milestone_future_miss_no_dis_mask_weshop.sh"
    )
    script = path.read_text()

    assert '"$PYTHON_BIN" -m recipe.graph_guide.main_graph_guide' in script
    assert 'model_path="${MODEL_PATH_1_5B}"' in script
    assert "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-0,1}" in script
    assert "algorithm.adv_estimator=migpo" in script
    assert "algorithm.gamma=0.95" in script
    assert "algorithm.migpo.step_advantage_w=10.0" in script
    assert "algorithm.migpo.mode=$mode" in script
    assert "algorithm.migpo.gamma=0.9" in script
    assert "algorithm.migpo.threshold=0.95" in script
    assert "++algorithm.graph_guide.env_adapter=migpo_native" in script
    assert "++algorithm.graph_guide.annotation_source=migpo_milestone" in script
    assert "++algorithm.graph_guide.migpo_reasonable_source=next_hit_segment" in script
    assert "++algorithm.graph_guide.migpo_threshold=0.95" in script
    assert "++algorithm.graph_guide.disadvantage_mode=advantage_cluster" in script
    assert "++algorithm.graph_guide.blame_source=milestone_future_miss" in script
    assert "data.max_prompt_length=4096" in script
    assert "env.env_name=Webshop" in script
    assert "env.webshop.use_small=True" in script
    assert "env.webshop.human_goals=False" in script
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=2" in script
    assert "trainer.n_gpus_per_node=2" in script
    assert "trainer.test_freq=5" in script
    assert "trainer.save_freq=50" in script


def test_fair_7b_weshop_migpo_graph_guide_v2_launcher_uses_native_milestone_credit():
    path = (
        GRAPH_GUIDE_DIR
        / "run_sh_fair"
        / "7B_new"
        / "seed0"
        / "weshop"
        / "migpo_graph_guide_v2_milestone_future_miss_no_dis_mask_weshop.sh"
    )
    script = path.read_text()

    assert '"$PYTHON_BIN" -m recipe.graph_guide.main_graph_guide' in script
    assert "setup_verl_7b_env weshop" in script
    assert 'model_path="${MODEL_PATH_7B}"' in script
    assert "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-0,1,2,3}" in script
    assert "algorithm.adv_estimator=migpo" in script
    assert "algorithm.gamma=0.95" in script
    assert "algorithm.migpo.step_advantage_w=10.0" in script
    assert "algorithm.migpo.mode=$mode" in script
    assert "algorithm.migpo.gamma=0.9" in script
    assert "algorithm.migpo.threshold=0.95" in script
    assert "++algorithm.graph_guide.env_adapter=migpo_native" in script
    assert "++algorithm.graph_guide.annotation_source=migpo_milestone" in script
    assert "++algorithm.graph_guide.migpo_reasonable_source=next_hit_segment" in script
    assert "++algorithm.graph_guide.migpo_threshold=0.95" in script
    assert "++algorithm.graph_guide.disadvantage_mode=advantage_cluster" in script
    assert "++algorithm.graph_guide.blame_source=milestone_future_miss" in script
    assert "data.max_prompt_length=4096" in script
    assert "env.env_name=Webshop" in script
    assert "env.webshop.use_small=True" in script
    assert "env.webshop.human_goals=False" in script
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=4" in script
    assert "trainer.n_gpus_per_node=4" in script
    assert "trainer.test_freq=5" in script
    assert "trainer.save_freq=50" in script
