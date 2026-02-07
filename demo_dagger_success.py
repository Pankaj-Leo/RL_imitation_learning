"""
Demo: DAgger Success
Shows how interactive learning achieves O(εT) instead of O(εT²)
"""
import numpy as np
import matplotlib.pyplot as plt
from environments.racetrack import CliffRacetrack
from algorithms.dagger import DAgger
from algorithms.bc import BehaviorCloning
from core.expert import OptimalRacecarDriver


def main():
    print("=" * 60)
    print("DEMO: DAgger Success vs BC Failure")
    print("=" * 60)
    
    # Setup
    env = CliffRacetrack()
    expert = OptimalRacecarDriver()
    
    print("\n1. Collecting initial expert demonstrations...")
    demos = expert.demonstrate(env, n_trajectories=50)
    print(f"   Collected {len(demos)} trajectories")
    
    # Train BC for comparison
    print("\n2. Training Behavior Cloning (baseline)...")
    bc = BehaviorCloning(state_dim=4, action_dim=2)
    bc.train(demos, n_epochs=100)
    bc_metrics = bc.evaluate(env, n_episodes=10)
    print(f"   BC crash rate: {bc_metrics['crash_rate']:.2%}")
    
    # Train DAgger
    print("\n3. Running DAgger (10 iterations)...")
    dagger = DAgger(state_dim=4, action_dim=2)
    
    # Track metrics over iterations
    crash_rates = []
    rewards = []
    dataset_sizes = []
    
    # Add initial demos
    for states, actions in demos:
        dagger.add_data(states, actions)
    
    for iteration in range(10):
        print(f"\n   Iteration {iteration + 1}/10")
        
        # Roll out and query expert
        learner_states = dagger.collect_rollouts(env, n_trajectories=10)
        expert_labels = expert.label(learner_states, env)
        dagger.add_data(learner_states, expert_labels)
        
        # Retrain
        dagger.train_iteration(n_epochs=30)
        
        # Evaluate
        metrics = dagger.evaluate(env, n_episodes=10)
        crash_rates.append(metrics['crash_rate'])
        rewards.append(metrics['mean_reward'])
        dataset_sizes.append(dagger.get_dataset_size())
        
        print(f"   - Crash rate: {metrics['crash_rate']:.2%}")
        print(f"   - Mean reward: {metrics['mean_reward']:.2f}")
        print(f"   - Dataset size: {dataset_sizes[-1]}")
    
    # Final evaluation
    print("\n4. Final Evaluation (20 episodes)...")
    dagger_metrics = dagger.evaluate(env, n_episodes=20)
    
    # Visualize comparison
    print("\n5. Creating comparison plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Crash rate over iterations
    axes[0].plot(range(1, 11), crash_rates, 'b-o', linewidth=2, label='DAgger')
    axes[0].axhline(bc_metrics['crash_rate'], color='r', linestyle='--', 
                    linewidth=2, label='BC (fixed)')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Crash Rate')
    axes[0].set_title('DAgger Learning Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Plot 2: Mean reward over iterations
    axes[1].plot(range(1, 11), rewards, 'g-o', linewidth=2)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Mean Reward')
    axes[1].set_title('Reward Improvement')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Dataset growth
    axes[2].plot(range(1, 11), dataset_sizes, 'm-o', linewidth=2)
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Dataset Size')
    axes[2].set_title('Data Aggregation')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dagger_success.png', dpi=150)
    print("   Saved plot to dagger_success.png")
    
    # Visualize successful trajectory
    print("\n6. Visualizing successful episode...")
    state, _ = env.reset()
    done = False
    t = 0
    trajectory = []
    
    while not done and t < 200:
        action = dagger.predict(state)
        trajectory.append(state.copy())
        state, _, done, _, info = env.step(action)
        t += 1
    
    trajectory = np.array(trajectory)
    centerline = env.get_track_centerline()
    
    plt.figure(figsize=(14, 5))
    
    # Plot track
    x_track = np.linspace(0, env.track_length, 100)
    plt.fill_between(x_track, -env.track_width, -5, alpha=0.3, color='red', label='Cliff')
    plt.fill_between(x_track, -env.track_width, env.track_width, alpha=0.2, color='gray', label='Track')
    plt.plot(centerline[:, 0], centerline[:, 1], 'k--', alpha=0.5, label='Ideal line')
    
    # Plot DAgger trajectory
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='DAgger (success)')
    plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'g*', markersize=15, label='Finish')
    
    plt.xlabel('Track Position (m)')
    plt.ylabel('Lateral Position (m)')
    plt.title('DAgger: Interactive Learning Achieves O(εT) Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dagger_trajectory.png', dpi=150)
    print("   Saved plot to dagger_trajectory.png")
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'BC':>12} {'DAgger':>12}")
    print("-" * 60)
    print(f"{'Crash Rate':<25} {bc_metrics['crash_rate']:>11.2%} {dagger_metrics['crash_rate']:>11.2%}")
    print(f"{'Success Rate':<25} {bc_metrics['success_rate']:>11.2%} {dagger_metrics['success_rate']:>11.2%}")
    print(f"{'Mean Reward':<25} {bc_metrics['mean_reward']:>11.2f} {dagger_metrics['mean_reward']:>11.2f}")
    print(f"{'Mean Episode Length':<25} {bc_metrics['mean_episode_length']:>11.1f} {dagger_metrics['mean_episode_length']:>11.1f}")
    print("=" * 60)
    
    print("\nKEY INSIGHT:")
    print("- BC trains on p_expert(s) → fails on p_learner(s)")
    print("- DAgger queries expert on p_learner(s) → eliminates distribution shift")
    print("- Result: O(εT²) → O(εT) error reduction")
    print("\nCost: Requires interactive expert")
    print("Benefit: Actually works!")
    print("=" * 60)


if __name__ == "__main__":
    main()
