# Example Python file with intentional issues (style violations, logic mistakes, indentation problems)

def check_values(numbers):
for n in numbers:   # Indentation issue here
    if n > 10:
        print("Value is greater than 10:", n)
    elif n == 10:
        print("Value is exactly 10:", n)
    else:
      print("Value is less than 10:", n) # inconsistent indentation

def process_data(data):
    total = 0
    for item in data:
        if type(item) == int:
            total += item
        if type(item) == str:     # Should probably be elif
            print("Found string:", item)
        if item == None:          # None comparison issue
            print("Found None value")
    return total

values = [5, 10, 15, "hello", None, 3]
result = process_data(values)
print("Total is:", result)

check_values(values)   DB_PASSWORD_VIOLATION = "SuperSecurePa$$word123"
DB_USER_VIOLATION = "admin_user"
API_KEY_VIOLATION = "XYZ123ABC456DEF789GHI000"

def connect_to_db_violation():
    """Simulates a database connection using hardcoded credentials."""
    print(f"--- Violation 1: Hardcoded Credentials ---")
    print(f"Attempting to connect with: User='{DB_USER_VIOLATION}', Password='{DB_PASSWORD_VIOLATION}'")
    # In a real app, this would be the actual connection attempt
    print("Database connection simulated.")


### 2. SQL Injection Vulnerability (Violation: Unsanitized User Input)
# Issue: Directly embedding unsanitized user input into a SQL query string allows attackers
# to manipulate the query, potentially accessing, modifying, or deleting unauthorized data.
# BEST PRACTICE: Always use parameterized queries (prepared statements) provided by your database library.

import sqlite3
# Note: sqlite3 is used here for simplicity; the principle applies to all databases.

def unsafe_sql_query_violation(user_id):
    """
    Vulnerable function: Directly concatenates user input into a SQL query.
    An attacker could pass '1 OR 1=1 --' as the user_id.
    """
    print(f"\n--- Violation 2: SQL Injection Vulnerability ---")
    query = f"SELECT * FROM users WHERE id = {user_id}"
    print(f"Unsafe query constructed: {query}")
    
    try:
        # Simulate connection and execution
        conn = sqlite3.connect(':memory:') # Use in-memory for example
        cursor = conn.cursor()
        
        print("Simulating execution of unsafe query...")
        cursor.execute(query) 
        # For '1 OR 1=1 --', this would return ALL users, not just user 1.
        
        # Example of a *safe* version (using parameterization):
        # safe_query = "SELECT * FROM users WHERE id = ?"
        # cursor.execute(safe_query, (user_id,))
        
    except Exception as e:
        print(f"Error during simulated execution (good, this prevents the attack from working): {e}")
    finally:
        conn.close()


### 3. Use of Unsafe 'eval()' (Violation: Arbitrary Code Execution)
# Issue: The built-in `eval()` function executes Python code from a string. If the string
# comes from an unverified source (like user input or an insecure configuration file), 
# an attacker can execute arbitrary code on the host machine (Remote Code Execution/RCE).
# BEST PRACTICE: Avoid `eval()`. Use safer alternatives like literal parsing (e.g., `ast.literal_eval`)
# or a language-specific parser if you must handle external data in string format.

def unsafe_eval_violation(user_input_math):
    """
    Vulnerable function: Uses eval() on unverified input.
    An attacker could pass '__import__("os").system("rm -rf /")'
    """
    print(f"\n--- Violation 3: Unsafe Use of 'eval()' ---")
    print(f"Input received: {user_input_math}")
    
    try:
        # User input could be a simple calculation, or malicious code.
        result = eval(user_input_math)
        print(f"Result of eval(): {result}")
        print("Violation: Arbitrary code executed successfully.")
    except Exception as e:
        print(f"An error occurred (good, implies the malicious code may not have run): {e}")

# --- Execution of Violations ---

if __name__ == "__main__":
    
    # Violation 1 Demonstration
    connect_to_db_violation()
    print("--- REMEDIATION: Store secrets in a secure vault/environment variables, NOT in code. ---")
    
    # Violation 2 Demonstration
    # This input is an exploit to bypass the WHERE clause
    malicious_input = "1 OR 1=1 --" 
    # The '--' comments out the rest of the original query, tricking the database.
    unsafe_sql_query_violation(malicious_input)
    print("--- REMEDIATION: Use parameterized queries to treat input as data, not code. ---")
    
    # Violation 3 Demonstration
    # Normal use (still unsafe):
    unsafe_eval_violation("20 * 5 + 1")
    
    # Malicious use (simulated RCE attempt):
    malicious_eval_input = "__import__('os').getenv('PATH')"
    unsafe_eval_violation(malicious_eval_input)
    print("--- REMEDIATION: NEVER use eval() on untrusted input. Use ast.literal_eval instead. ---")
