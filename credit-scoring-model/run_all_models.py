import subprocess
import sys
import time
from datetime import datetime


def print_header(title):
    print("\n" + "=" * 80)
    print(f"{title.center(80)}")
    print("=" * 80 + "\n")


def print_section(title):
    print("\n" + "-" * 80)
    print(f"{title}")
    print("-" * 80 + "\n")


def run_script(script_name, description):
    print_section(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ {script_name} completed successfully in {elapsed_time:.2f} seconds")
        return True, elapsed_time
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ {script_name} failed after {elapsed_time:.2f} seconds")
        print(f"Error: {e}")
        return False, elapsed_time
    except FileNotFoundError:
        print(f"\n✗ Script not found: {script_name}")
        return False, 0


def main():
    print_header("CREDIT SCORING MODEL - COMPLETE PIPELINE EXECUTION")
    
    print("Starting comprehensive model training and evaluation pipeline...")
    print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    pipeline_start_time = time.time()
    
    scripts = [
        ("train_logistic_regression.py", "Logistic Regression Baseline Model"),
        ("train_tree_models.py", "Decision Tree & Random Forest Comparison"),
        ("audit_baseline_model.py", "Baseline Model Audit & Analysis"),
        ("optimize_random_forest.py", "Random Forest Hyperparameter Optimization")
    ]
    
    results = []
    
    for script_name, description in scripts:
        success, elapsed = run_script(script_name, description)
        results.append({
            'script': script_name,
            'description': description,
            'success': success,
            'time': elapsed
        })
        
        if not success:
            print(f"\n⚠️  Warning: {script_name} encountered an error but continuing...")
        
        time.sleep(1)
    
    total_time = time.time() - pipeline_start_time
    
    print_header("PIPELINE EXECUTION SUMMARY")
    
    print(f"{'Script':<40} {'Status':<15} {'Time (s)':<15}")
    print("-" * 80)
    
    successful_count = 0
    failed_count = 0
    
    for result in results:
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        status_color = status
        print(f"{result['script']:<40} {status_color:<15} {result['time']:>10.2f}")
        
        if result['success']:
            successful_count += 1
        else:
            failed_count += 1
    
    print("-" * 80)
    print(f"Total scripts executed: {len(results)}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    print("\n" + "=" * 80)
    
    if failed_count == 0:
        print("🎉 ALL MODELS EXECUTED SUCCESSFULLY!")
        print("=" * 80)
        print("\nYour credit scoring pipeline is fully operational.")
        print("\nKey Results:")
        print("  • Logistic Regression baseline trained and evaluated")
        print("  • Decision Tree and Random Forest models compared")
        print("  • Baseline model audited with feature importance analysis")
        print("  • Random Forest hyperparameters optimized with threshold tuning")
        print("\nNext Steps:")
        print("  • Review the detailed output above for model performance metrics")
        print("  • Choose the best model based on ROC-AUC scores")
        print("  • Select optimal probability threshold based on business requirements")
    else:
        print("⚠️  PIPELINE COMPLETED WITH ERRORS")
        print("=" * 80)
        print(f"\n{failed_count} script(s) failed. Please review the errors above.")
    
    print(f"\nPipeline finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
