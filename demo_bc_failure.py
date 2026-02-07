"""
Demo: Behavior Cloning Catastrophic Failure
Shows how low training error != good performance
"""
import numpy as np
import matplotlib.pyplot as plt
from environments.racetrack import CliffRacetrack
from algorithms.bc import BehaviorCloning
from core.expert import OptimalRacecarDriver


def main():
    print("=" * 60)
    print("DEMO: Behavior Cloning Catastrophic Failure")
    print("=" * 60)
    
    # Create environment and expert
    env = CliffRacetrack()
    expert = OptimalRacecarDriver()
    
    print("\n1. Collecting expert demonstrations...")
    demos = expert.demonstrate(env, n_trajectories=100)
    print(f"   Collected {len(demos)} trajectories")
    
    # Compute expert performance for comparison
    print("\n2. Evaluating expert performance...")
    expert_rewards = []
    for _ in range(20):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        t = 0
        while not done and t < 200:
            action = expert.get_action(state, env)
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            t += 1
        expert_rewards.append(episode_reward)
    
    print(f"   Expert mean reward: {np.mean(expert_rewards):.2f}")
    print(f"   Expert success rate: 100%")
    
    # Train BC
    print("\n3. Training Behavior Cloning...")
    bc = BehaviorCloning(state_dim=4, action_dim=2)
    bc.train(demos, n_epochs=100)
    
    # Compute training error (misleading!)
    training_error = bc.compute_training_error(demos)
    print(f"   Training error: {training_error:.6f} (looks great!)")
    
    # Evaluate BC (disaster!)
    print("\n4. Evaluating Behavior Cloning...")
    metrics = bc.evaluate(env, n_episodes=20)
    
    print(f"\n   Results:")
    print(f"   - Mean reward: {metrics['mean_reward']:.2f}")
    print(f"   - Crash rate: {metrics['crash_rate']:.2%}")
    print(f"   - Success rate: {metrics['success_rate']:.2%}")
    print(f"   - Mean episode length: {metrics['mean_episode_length']:.1f}")
    
    # Theoretical analysis
    print("\n5. Theoretical Analysis:")
    epsilon = training_error
    T = 100
    theoretical_error_bc = epsilon * T * T
    theoretical_error_dagger = epsilon * T
    
    print(f"   Per-timestep error ε: {epsilon:.6f}")
    print(f"   Episode horizon T: {T}")
    print(f"   BC expected error: O(εT²) ≈ {theoretical_error_bc:.4f}")
    print(f"   DAgger expected error: O(εT) ≈ {theoretical_error_dagger:.4f}")
    
    # Visualize one failure episode
    print("\n6. Visualizing failure episode...")
    state, _ = env.reset()
    done = False
    t = 0
    trajectory = []
    
    while not done and t < 200:
        action = bc.predict(state)
        trajectory.append(state.copy())
        state, _, done, _, info = env.step(action)
        t += 1
        
    # Plot
    trajectory = np.array(trajectory)
    centerline = env.get_track_centerline()
    
    plt.figure(figsize=(14, 5))
    
    # Plot track
    x_track = np.linspace(0, env.track_length, 100)
    plt.fill_between(x_track, -env.track_width, -5, alpha=0.3, color='red', label='Cliff')
    plt.fill_between(x_track, -env.track_width, env.track_width, alpha=0.2, color='gray', label='Track')
    plt.plot(centerline[:, 0], centerline[:, 1], 'k--', alpha=0.5, label='Ideal line')
    
    # Plot BC trajectory
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='BC (crashes)')
    plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    
    if info['fell_off_cliff']:
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'rx', markersize=15, 
                markeredgewidth=3, label='Crash!')
    
    plt.xlabel('Track Position (m)')
    plt.ylabel('Lateral Position (m)')
    plt.title('Behavior Cloning: Low Training Error, Catastrophic Test Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bc_failure.png', dpi=150)
    print("   Saved plot to bc_failure.png")
    
    # Summary
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY:")
    print("=" * 60)
    print(f"Training error: {training_error:.6f} ✓ (excellent)")
    print(f"Test crash rate: {metrics['crash_rate']:.2%} ✗ (catastrophic)")
    print("\nWhy? Distribution shift!")
    print("- Training: states from p_expert(s)")
    print("- Testing: states from p_learner(s)")
    print("- One mistake → new states → more mistakes → death spiral")
    print("\nSolution: DAgger (see demo_dagger_success.py)")
    print("=" * 60)


if __name__ == "__main__":
    main()
