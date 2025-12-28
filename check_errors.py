"""
Error Checker for Face Recognition Project

This script checks for common errors in all project files.

Project Information:
- Project ID: 22
- Title: Face Recognition Dataset
- Category: Image Data
- Technologies: PNG, JPG, NumPy, OpenCV, Face Recognition

Contact Information:
RSK World
Founder: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277
Address: Nutanhat, Mongolkote, Purba Burdwan, West Bengal, India, 713147
Website: https://rskworld.in/
Year: 2026
"""

import os
import sys
import ast
import importlib.util


def check_syntax(filepath):
    """Check Python file syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def check_imports(filepath):
    """Check if imports are valid (basic check)."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Check for common import issues
        issues = []
        
        # Check for COLOR_RGB2RGB typo (exclude this checker file itself)
        if 'COLOR_RGB2RGB' in code and 'check_errors.py' not in filepath:
            issues.append("Found invalid COLOR_RGB2RGB (should be COLOR_BGR2RGB or no conversion needed)")
        
        # Check for missing try-except around optional imports
        if 'from scripts.advanced_features import' in code and 'try:' not in code:
            # This is okay, just a note
            pass
        
        return len(issues) == 0, issues
    except Exception as e:
        return False, [str(e)]


def check_file_structure():
    """Check if required files and directories exist."""
    required_files = [
        'config.py',
        'requirements.txt',
        'README.md',
        'train_model.py',
        'scripts/__init__.py',
        'scripts/load_dataset.py',
        'scripts/preprocess.py',
        'scripts/recognize_faces.py',
    ]
    
    required_dirs = [
        'data',
        'data/train',
        'scripts',
        'models',
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    for dir in required_dirs:
        if not os.path.exists(dir):
            missing_dirs.append(dir)
    
    return missing_files, missing_dirs


def main():
    """Run all checks."""
    print("=" * 60)
    print("Face Recognition Project - Error Checker")
    print("=" * 60)
    print(f"Project: Face Recognition Dataset (ID: 22)")
    print(f"RSK World - https://rskworld.in/")
    print("=" * 60)
    print()
    
    errors_found = False
    
    # Check file structure
    print("Checking file structure...")
    missing_files, missing_dirs = check_file_structure()
    
    if missing_files:
        print(f"[ERROR] Missing files: {', '.join(missing_files)}")
        errors_found = True
    else:
        print("[OK] All required files present")
    
    if missing_dirs:
        print(f"[ERROR] Missing directories: {', '.join(missing_dirs)}")
        errors_found = True
    else:
        print("[OK] All required directories present")
    
    print()
    
    # Check Python files
    print("Checking Python files...")
    python_files = []
    
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                python_files.append(filepath)
    
    syntax_errors = []
    import_issues = []
    
    for filepath in python_files:
        # Check syntax
        is_valid, error = check_syntax(filepath)
        if not is_valid:
            syntax_errors.append((filepath, error))
            errors_found = True
        
        # Check imports
        is_valid, issues = check_imports(filepath)
        if not is_valid:
            import_issues.append((filepath, issues))
            errors_found = True
    
    if syntax_errors:
        print("[ERROR] Syntax errors found:")
        for filepath, error in syntax_errors:
            print(f"  {filepath}: {error}")
    else:
        print("[OK] No syntax errors")
    
    if import_issues:
        print("[ERROR] Import issues found:")
        for filepath, issues in import_issues:
            print(f"  {filepath}: {', '.join(issues)}")
    else:
        print("[OK] No import issues")
    
    print()
    
    # Summary
    print("=" * 60)
    if errors_found:
        print("[ERROR] Some issues were found. Please review above.")
        return 1
    else:
        print("[OK] All checks passed! No errors found.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

