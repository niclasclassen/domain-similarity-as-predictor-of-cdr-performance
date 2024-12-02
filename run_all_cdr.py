import subprocess

# Define the list of domains
domains = ["Musical_Instruments", "Software", "Video_Games"]  # Add your domains here

# Loop through all combinations of source and target domains
for source_domain in domains:
    for target_domain in domains:
        if source_domain != target_domain:
            # Run the first Python script
            print(f"Running get_domain_rec.py with source: {source_domain} and target: {target_domain}")
            subprocess.run(["python", "get_domain_rec.py", "--source_domain", source_domain, "--target_domain", target_domain])

            # Run the second Python script
            print(f"Running run_recbole_cdr.py with model EMCDR")
            subprocess.run(["python", "run_recbole_cdr.py", "--model", "'EMCDR'", "--source_domain", source_domain, "--target_domain", target_domain])

            print(f"Finished processing source: {source_domain} and target: {target_domain}")
