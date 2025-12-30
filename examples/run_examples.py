#!/usr/bin/env python3
"""
EMTL Examples Runner
====================

This script runs comprehensive examples of the EMTL solver,
demonstrating various parameter combinations and their results.

Run with: python examples/run_examples.py
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emtl_solver import solve_emtl, GraphParameters, SolverStatus


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    print(f"\n{char * 70}")
    print(f"  {text}")
    print(f"{char * 70}\n")


def print_subheader(text: str):
    """Print a formatted subheader."""
    print(f"\n{'─' * 50}")
    print(f"  {text}")
    print(f"{'─' * 50}\n")


def run_single_example(m: int, n: int, k: int, t: int, 
                       description: str = "", 
                       save_fig: bool = True) -> dict:
    """
    Run a single EMTL example and return results.
    """
    print_subheader(f"Example: G(m={m}, n={n}, k={k}, t={t})")
    
    if description:
        print(f"Description: {description}")
    
    # Calculate expected sizes
    params = GraphParameters(m=m, n=n, k=k, t=t)
    print(f"Expected: |V|={params.num_vertices}, |E|={params.num_edges}, "
          f"Labels={params.total_labels}")
    
    # Validate
    is_valid, error = params.validate()
    if not is_valid:
        print(f"⚠️ Invalid parameters: {error}")
        return {'valid': False, 'error': error}
    
    # Solve
    fig_path = f"images/output/emtl_m{m}_n{n}_k{k}_t{t}.png" if save_fig else None
    
    start_time = time.time()
    result = solve_emtl(
        m=m, n=n, k=k, t=t,
        timeout=60,
        visualize=False,
        save_fig=fig_path,
        verbose=True
    )
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\n{'─' * 30}")
    print(f"RESULT: {result.status.value.upper()}")
    if result.exists:
        print(f"Magic constant: k = {result.magic_constant}")
    print(f"Total time: {elapsed:.3f}s")
    
    if save_fig and result.exists:
        print(f"Figure saved: {fig_path}")
    
    return {
        'valid': True,
        'm': m, 'n': n, 'k': k, 't': t,
        'status': result.status,
        'exists': result.exists,
        'magic_constant': result.magic_constant,
        'solve_time': result.solve_time,
        'vertices': params.num_vertices,
        'edges': params.num_edges,
    }


def run_comprehensive_examples():
    """
    Run a comprehensive set of examples covering various cases.
    """
    print_header("EMTL SOLVER - COMPREHENSIVE EXAMPLES", "█")
    
    # Create output directory
    os.makedirs("images/output", exist_ok=True)
    
    # Define example categories
    examples = {
        "Minimal and Small Graphs": [
            (1, 1, 1, 0, "Minimal graph with no B-C edges"),
            (1, 1, 1, 1, "Smallest connected graph"),
            (1, 2, 1, 1, "Small asymmetric graph"),
            (2, 2, 2, 1, "Basic example"),
        ],
        "Symmetric Graphs": [
            (2, 2, 2, 2, "Symmetric (2,2,2,2)"),
            (3, 3, 3, 3, "Symmetric (3,3,3,3)"),
            (4, 4, 4, 4, "Symmetric (4,4,4,4)"),
        ],
        "Varying Regularity": [
            (2, 3, 2, 0, "No B-C edges (t=0)"),
            (2, 3, 2, 1, "Low regularity (t=1)"),
            (2, 3, 2, 2, "Medium regularity (t=2)"),
            (2, 3, 2, 3, "Maximum regularity (t=n)"),
        ],
        "Asymmetric Graphs": [
            (1, 3, 1, 2, "Small A and D"),
            (3, 2, 3, 2, "Large A and D"),
            (1, 4, 2, 2, "Very small A"),
            (4, 2, 1, 2, "Large A, small D"),
        ],
        "Larger Graphs": [
            (3, 4, 3, 3, "Medium-large graph"),
            (4, 5, 4, 4, "Larger graph"),
            (5, 5, 5, 5, "Large symmetric graph"),
        ],
    }
    
    # Track results
    all_results = []
    
    # Run examples by category
    for category, example_list in examples.items():
        print_header(category, "═")
        
        for example in example_list:
            if len(example) == 5:
                m, n, k, t, desc = example
            else:
                m, n, k, t = example
                desc = ""
            
            result = run_single_example(m, n, k, t, desc)
            all_results.append(result)
    
    # Print summary table
    print_header("RESULTS SUMMARY", "█")
    
    print(f"{'Parameters':<20} {'|V|':>5} {'|E|':>5} {'Labels':>7} "
          f"{'Status':>12} {'Magic k':>8} {'Time':>8}")
    print("─" * 75)
    
    found_count = 0
    infeasible_count = 0
    timeout_count = 0
    
    for r in all_results:
        if not r['valid']:
            continue
        
        params_str = f"({r['m']},{r['n']},{r['k']},{r['t']})"
        
        if r['status'] == SolverStatus.FOUND:
            status_str = "✓ Found"
            magic_str = str(r['magic_constant'])
            found_count += 1
        elif r['status'] == SolverStatus.INFEASIBLE:
            status_str = "✗ None"
            magic_str = "—"
            infeasible_count += 1
        else:
            status_str = "⏱ Timeout"
            magic_str = "?"
            timeout_count += 1
        
        time_str = f"{r['solve_time']:.2f}s"
        
        print(f"{params_str:<20} {r['vertices']:>5} {r['edges']:>5} "
              f"{r['vertices']+r['edges']:>7} {status_str:>12} "
              f"{magic_str:>8} {time_str:>8}")
    
    print("─" * 75)
    print(f"\nTotal examples: {len(all_results)}")
    print(f"  ✓ EMTL Found: {found_count}")
    print(f"  ✗ No EMTL: {infeasible_count}")
    print(f"  ⏱ Timeout: {timeout_count}")
    
    return all_results


def run_stress_test():
    """
    Run stress tests with larger graphs.
    """
    print_header("STRESS TEST - LARGER GRAPHS", "█")
    
    stress_cases = [
        (5, 5, 5, 5),
        (4, 6, 4, 5),
        (6, 6, 6, 6),
        (3, 7, 3, 5),
    ]
    
    for m, n, k, t in stress_cases:
        print_subheader(f"Stress Test: G(m={m}, n={n}, k={k}, t={t})")
        
        params = GraphParameters(m=m, n=n, k=k, t=t)
        print(f"Graph size: |V|={params.num_vertices}, |E|={params.num_edges}")
        print(f"Labels to assign: {params.total_labels}")
        
        result = solve_emtl(
            m=m, n=n, k=k, t=t,
            timeout=120,
            visualize=False,
            verbose=True
        )
        
        if result.exists:
            print(f"✓ Found EMTL with k={result.magic_constant} in {result.solve_time:.2f}s")
        else:
            print(f"Status: {result.status.value}")


def run_boundary_tests():
    """
    Test boundary conditions.
    """
    print_header("BOUNDARY CONDITION TESTS", "█")
    
    boundary_cases = [
        # t = 0 (no B-C edges, disconnected graph)
        (2, 3, 2, 0, "Disconnected graph (t=0)"),
        (3, 3, 3, 0, "Larger disconnected graph"),
        
        # t = n (complete B-C bipartite)
        (2, 3, 2, 3, "Complete B-C (t=n)"),
        (2, 4, 2, 4, "Larger complete B-C"),
        
        # Single vertex partitions
        (1, 1, 1, 1, "All singletons"),
        (1, 5, 1, 3, "Only B/C have multiple vertices"),
        
        # Extreme aspect ratios
        (5, 2, 5, 2, "Wide graph (large A, D)"),
        (1, 5, 1, 5, "Narrow graph (small A, D)"),
    ]
    
    for case in boundary_cases:
        m, n, k, t, desc = case
        run_single_example(m, n, k, t, desc, save_fig=False)


def interactive_mode():
    """
    Run in interactive mode, allowing user to input parameters.
    """
    print_header("INTERACTIVE MODE", "█")
    
    print("Enter graph parameters (or 'q' to quit)")
    print("Format: m n k t")
    print()
    
    while True:
        try:
            user_input = input("Parameters (m n k t): ").strip()
            
            if user_input.lower() == 'q':
                print("Goodbye!")
                break
            
            parts = user_input.split()
            if len(parts) != 4:
                print("Please enter exactly 4 numbers: m n k t")
                continue
            
            m, n, k, t = map(int, parts)
            run_single_example(m, n, k, t)
            
        except ValueError:
            print("Invalid input. Please enter integers.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    """
    Main entry point for examples runner.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="EMTL Solver Examples Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_examples.py              # Run comprehensive examples
  python run_examples.py --stress     # Run stress tests
  python run_examples.py --boundary   # Run boundary tests
  python run_examples.py --interactive # Interactive mode
  python run_examples.py --all        # Run all tests
        """
    )
    
    parser.add_argument('--stress', action='store_true',
                       help='Run stress tests with larger graphs')
    parser.add_argument('--boundary', action='store_true',
                       help='Run boundary condition tests')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Run all tests')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.all:
        run_comprehensive_examples()
        run_stress_test()
        run_boundary_tests()
    elif args.stress:
        run_stress_test()
    elif args.boundary:
        run_boundary_tests()
    else:
        run_comprehensive_examples()


if __name__ == "__main__":
    main()

