import pandas as pd
from pathlib import Path

def check_simulation_status(input_dir, output_dir, output_csv):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    results = []
    
    input_files = list(input_path.glob("*_s2b_cropped.tif"))
    
    for file in input_files:
        product_id = file.name.split('_')[0]
        
        simulated_file = next(output_path.rglob(f"simulated_*{file.name}"), None)
        
        if simulated_file:
            status = "SUCCESS"
            sim_path = str(simulated_file.absolute())
        else:
            status = "FAILED"
            sim_path = "NOT_FOUND"
        
        results.append({
            "product_id": product_id,
            "simulation_status": status,
            "simulation_path": sim_path
        })
    
    df = pd.DataFrame(results)
    
    df['product_id'] = pd.to_numeric(df['product_id'], errors='coerce')
    df = df.sort_values('product_id')
    
    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"Report generated at : {output_csv}")
    print(f"Total processed : {len(df)} | Success : {len(df[df['simulation_status'] == 'SUCCESS'])}")

if __name__ == "__main__":
    INPUT_DIR = "/shared/projects/phisat2/data/interim/s2b_croped"
    OUTPUT_DIR = "/shared/projects/phisat2/data/interim/s2b_simulated"
    OUTPUT_CSV = "/shared/projects/phisat2/data/index/simulation_status_report.csv"
    
    check_simulation_status(INPUT_DIR, OUTPUT_DIR, OUTPUT_CSV)