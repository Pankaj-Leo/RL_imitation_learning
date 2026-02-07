"""
Getting Started with Imitation Learning
Run this script to understand the full progression:
BC fails → DAgger works → AggreVaTe is better
"""
import numpy as np
import matplotlib.pyplot as plt
from environments.racetrack import CliffRacetrack
from algorithms.bc import BehaviorCloning
from algorithms.dagger import DAgger
from algorithms.aggravate import AggreVaTe
from core.expert import OptimalRacecarDriver


def run_all_comparisons():
    """Run complete comparison of all three algorithms"""
    
    print("=" * 70)
    print("IMITATION LEARNING: FROM THEORY TO PRACTICE")
    print("=" * 70)
    print("\nThis demo shows:")
    print("1. BC: Low training error, catastrophic test failure (O(εT²))")
    print("2. DAgger: Interactive learning eliminates distribution shift (O(εT))")
    print("3. AggreVaTe: Value-aware learning handles edge cases (O(εT))")
    print("=" * 70)
    
    # Setup
    env = CliffRacetrack()
    expert = OptimalRacecarDriver()
    
    # Collect initial demonstrations
    print("\n📦 Collecting expert demonstrations...")
    demos = expert.demonstrate(env, n_trajectories=50)
    print(f"   ✓ Collected {len(demos)} trajectories")
    
    # Expert baseline
    print("\n🎯 Evaluating expert (baseline)...")
    expert_metrics = evaluate_policy(env, expert, n_episodes=10)
    print(f"   Expert success rate: {expert_metrics['success_rate']:.1%}")
    
    # Algorithm 1: Behavior Cloning
    print("\n" + "=" * 70)
    print("ALGORITHM 1: BEHAVIOR CLONING")
    print("=" * 70)
    print("Supervised learning on expert states")
    print("Expected: Low training error, high test error due to O(εT²) compounding")
    
    bc = BehaviorCloning(state_dim=4, action_dim=2)
    print("\n🔧 Training BC...")
    bc.train(demos, n_epochs=100)
    
    train_error = bc.compute_training_error(demos)
    bc_metrics = bc.evaluate(env, n_episodes=20)
    
    print(f"\n📊 Results:")
    print(f"   Training error: {train_error:.6f} ← Looks great!")
    print(f"   Success rate: {bc_metrics['success_rate']:.1%} ← Terrible!")
    print(f"   Why? Distribution shift: p_train ≠ p_test")
    
    # Algorithm 2: DAgger
    print("\n" + "=" * 70)
    print("ALGORITHM 2: DAGGER")
    print("=" * 70)
    print("Interactive learning: query expert on learner's states")
    print("Expected: O(εT) error, actually works")
    
    dagger = DAgger(state_dim=4, action_dim=2)
    print("\n🔧 Running DAgger (5 iterations)...")
    
    dagger_history = {'crash_rate': [], 'success_rate': []}
    
    # Initialize
    for states, actions in demos[:10]:
        dagger.add_data(states, actions)
    dagger.train_iteration(n_epochs=50)
    
    for i in range(5):
        # Roll out and query
        learner_states = dagger.collect_rollouts(env, n_trajectories=10)
        expert_labels = expert.label(learner_states, env)
        dagger.add_data(learner_states, expert_labels)
        
        # Retrain
        dagger.train_iteration(n_epochs=30)
        
        # Evaluate
        metrics = dagger.evaluate(env, n_episodes=10)
        dagger_history['crash_rate'].append(metrics['crash_rate'])
        dagger_history['success_rate'].append(metrics['success_rate'])
        
        print(f"   Iter {i+1}: Success rate = {metrics['success_rate']:.1%}")
    
    dagger_metrics = dagger.evaluate(env, n_episodes=20)
    print(f"\n📊 Final Results:")
    print(f"   Success rate: {dagger_metrics['success_rate']:.1%} ← Much better!")
    print(f"   Key insight: Query expert on p_learner(s), not p_expert(s)")
    
    # Algorithm 3: AggreVaTe
    print("\n" + "=" * 70)
    print("ALGORITHM 3: AGGRAVATE")
    print("=" * 70)
    print("Value-aware: learn which mistakes are actually dangerous")
    print("Expected: Handles edge cases better than DAgger")
    
    aggravate = AggreVaTe(state_dim=4, action_dim=2)
    print("\n🔧 Running AggreVaTe (5 iterations)...")
    
    aggravate_history = {'crash_rate': [], 'success_rate': []}
    
    # Initialize
    for states, actions in demos[:10]:
        advantages = np.zeros(len(states))
        aggravate.add_data(states, actions, advantages)
    aggravate.train_policy_with_advantages(n_epochs=50)
    
    for i in range(5):
        # Roll out and query for advantages
        learner_states = aggravate.collect_rollouts(env, n_trajectories=10)
        expert_actions = expert.label(learner_states, env)
        expert_advantages = expert.compute_advantages(learner_states, expert_actions, env)
        aggravate.add_data(learner_states, expert_actions, expert_advantages)
        
        # Train Q-network and policy
        all_states = np.concatenate(aggravate.states_buffer, axis=0)
        all_actions = np.concatenate(aggravate.actions_buffer, axis=0)
        all_advantages = np.concatenate(aggravate.advantages_buffer, axis=0)
        aggravate.train_q_network(all_states, all_actions, all_advantages, n_epochs=30)
        aggravate.train_policy_with_advantages(n_epochs=30)
        
        # Evaluate
        metrics = aggravate.evaluate(env, n_episodes=10)
        aggravate_history['crash_rate'].append(metrics['crash_rate'])
        aggravate_history['success_rate'].append(metrics['success_rate'])
        
        print(f"   Iter {i+1}: Success rate = {metrics['success_rate']:.1%}")
    
    aggravate_metrics = aggravate.evaluate(env, n_episodes=20)
    print(f"\n📊 Final Results:")
    print(f"   Success rate: {aggravate_metrics['success_rate']:.1%}")
    print(f"   Key insight: Cost-sensitive classification based on Q-values")
    
    # Create comparison plots
    print("\n📈 Creating comparison plots...")
    
    fig = plt.figure(figsize=(16, 5))
    
    # Plot 1: Final comparison
    ax1 = plt.subplot(1, 3, 1)
    algorithms = ['Expert', 'BC', 'DAgger', 'AggreVaTe']
    success_rates = [
        expert_metrics['success_rate'],
        bc_metrics['success_rate'],
        dagger_metrics['success_rate'],
        aggravate_metrics['success_rate']
    ]
    colors = ['green', 'red', 'blue', 'purple']
    bars = ax1.bar(algorithms, success_rates, color=colors, alpha=0.7)
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Final Performance Comparison')
    ax1.set_ylim([0, 1.1])
    ax1.axhline(y=0.9, color='k', linestyle='--', alpha=0.3, label='90% threshold')
    ax1.legend()
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: DAgger learning curve
    ax2 = plt.subplot(1, 3, 2)
    iterations = list(range(1, 6))
    ax2.plot(iterations, dagger_history['success_rate'], 'b-o', linewidth=2, 
            markersize=8, label='DAgger')
    ax2.axhline(y=bc_metrics['success_rate'], color='r', linestyle='--', 
               linewidth=2, label='BC (fixed)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('DAgger Learning Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    # Plot 3: AggreVaTe learning curve
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(iterations, aggravate_history['success_rate'], 'purple', marker='s', 
            linewidth=2, markersize=8, label='AggreVaTe')
    ax3.plot(iterations, dagger_history['success_rate'], 'b--', linewidth=2, 
            alpha=0.5, label='DAgger')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Success Rate')
    ax3.set_title('AggreVaTe vs DAgger')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('complete_comparison.png', dpi=150)
    print("   ✓ Saved to complete_comparison.png")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Algorithm':<15} {'Success Rate':<15} {'Error Bound':<15} {'Key Insight'}")
    print("-" * 70)
    print(f"{'Expert':<15} {expert_metrics['success_rate']:.1%}{'':^12} {'--':<15} Optimal baseline")
    print(f"{'BC':<15} {bc_metrics['success_rate']:.1%}{'':^12} {'O(εT²)':<15} Distribution shift kills it")
    print(f"{'DAgger':<15} {dagger_metrics['success_rate']:.1%}{'':^12} {'O(εT)':<15} Query on learner states")
    print(f"{'AggreVaTe':<15} {aggravate_metrics['success_rate']:.1%}{'':^12} {'O(εT)':<15} Cost-sensitive learning")
    print("=" * 70)
    
    print("\n🎓 KEY LESSONS:")
    print("   1. Low training error ≠ good performance (BC)")
    print("   2. Interactive learning fixes distribution shift (DAgger)")
    print("   3. Not all mistakes are equal (AggreVaTe)")
    print("   4. All algorithms solve min-max game in different ways")
    
    print("\n📚 NEXT STEPS:")
    print("   • Read core/policies.py for network architectures")
    print("   • Check algorithms/*.py for implementation details")
    print("   • Run experiments/* for deeper analysis")
    print("   • See README.md for mathematical details")
    
    print("\n✨ You now understand imitation learning from theory to practice!")
    print("=" * 70)


def evaluate_policy(env, policy_or_expert, n_episodes=10):
    """Helper to evaluate any policy"""
    from core.expert import OptimalRacecarDriver
    
    is_expert = isinstance(policy_or_expert, OptimalRacecarDriver)
    
    rewards = []
    successes = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        t = 0
        
        while not done and t < 200:
            if is_expert:
                action = policy_or_expert.get_action(state, env)
            else:
                action = policy_or_expert.predict(state)
                
            state, reward, done, _, info = env.step(action)
            episode_reward += reward
            t += 1
            
        rewards.append(episode_reward)
        successes.append(not info.get('fell_off_cliff', False))
    
    return {
        'mean_reward': np.mean(rewards),
        'success_rate': np.mean(successes),
        'crash_rate': 1.0 - np.mean(successes)
    }


if __name__ == "__main__":
    run_all_comparisons()
