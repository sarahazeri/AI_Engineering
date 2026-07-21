# test_tools.py
import os
from tools import read_file, write_file, edit_file, list_files, run_command

def run_quick_test():
    test_filename = "sample_test.txt"
    
    print("=== 1. Testing write_file ===")
    res_write = write_file(test_filename, "Hello! This is line 1.\nThis is line 2.")
    print("Result:", res_write)
    
    print("\n=== 2. Testing read_file ===")
    res_read = read_file(test_filename)
    print("Content read from file:\n", res_read)
    
    print("\n=== 3. Testing edit_file ===")
    res_edit = edit_file(test_filename, "line 2", "modified line 2")
    print("Result:", res_edit)
    print("Updated Content:\n", read_file(test_filename))
    
    print("\n=== 4. Testing list_files ===")
    res_list = list_files(".")
    print("Files in current directory:\n", res_list)
    
    print("\n=== 5. Testing run_command ===")
    # Running a simple python print command via shell
    res_cmd = run_command("python -c \"print('Hello from subprocess!')\"")
    print("Command Output:", res_cmd)
    
    # Cleanup test file
    if os.path.exists(test_filename):
        os.remove(test_filename)
        print(f"\n[Cleanup] Removed temporary file '{test_filename}'.")

if __name__ == "__main__":
    run_quick_test()