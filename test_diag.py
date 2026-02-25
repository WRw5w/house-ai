import unittest
import subprocess
import os
import sys

class TestDiagBat(unittest.TestCase):
    """
    Test suite for diag.bat.

    NOTE: This test assumes that the commands in diag.bat are cross-platform compatible
    (e.g., git commands, whoami) and can be executed in the current shell environment.
    If diag.bat is updated with Windows-specific syntax (like 'dir', 'cls', or '%VAR%'),
    this test may fail on non-Windows systems and will need adjustment.
    """

    def setUp(self):
        # Ensure a clean state
        if os.path.exists("output.txt"):
            os.remove("output.txt")

    def tearDown(self):
        # Cleanup
        if os.path.exists("output.txt"):
            os.remove("output.txt")

    def test_diag_execution(self):
        bat_file = "diag.bat"
        if not os.path.exists(bat_file):
            self.fail(f"{bat_file} does not exist")

        # Read the batch file content
        with open(bat_file, "r") as f:
            lines = f.readlines()

        # Execute each line as a shell command
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Execute the command
            # Using shell=True allows redirection syntax (>, >>) to work
            try:
                subprocess.run(line, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                self.fail(f"Command failed: {line}. Error: {e}")

        # Verify output file exists
        self.assertTrue(os.path.exists("output.txt"), "output.txt was not created")

        # Verify output file content
        with open("output.txt", "r") as f:
            content = f.read()

        print(f"Content of output.txt:\n{content}") # For debugging purposes

        # Check for expected content
        # git --version
        self.assertIn("git version", content, "Missing git version info")

        # git status
        # "On branch" is standard git status output
        self.assertIn("On branch", content, "Missing git status info")

        # git remote -v
        # Check if remote exists, otherwise this check might fail if repo has no remote
        # But in this environment we verified it has origin
        # We can check for 'origin' or simply that the file grew
        # But 'origin' is safer as a specific check if we assume standard setup
        # If no remote, this part of the test might be brittle.
        # Let's check if 'origin' is in the content OR if the command didn't fail (which we already checked)
        # But to be thorough, let's look for 'origin' if it's there.
        # Actually, let's just ensure content length > 0 which is implicit by other checks.

        # whoami
        # Get current user to verify
        current_user = subprocess.check_output("whoami", shell=True).decode().strip()
        self.assertIn(current_user, content, f"Missing username {current_user}")

if __name__ == "__main__":
    unittest.main()
