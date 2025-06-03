import os
import json
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(filename='visualization.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Ensure the output directory exists
output_dir = 'visualizations'
os.makedirs(output_dir, exist_ok=True)

# Input JSON data
data = {
    "metrics": {
        "Open ALL RRR Defects": {
            "ATLS": [
                {"version": "25.1", "value": 42, "status": "MEDIUM RISK", "trend": "→"},
                {"version": "25.2", "value": 21, "status": "MEDIUM RISK", "trend": "↓ (50.0%)"},
                {"version": "25.3", "value": 24, "status": "ON TRACK", "trend": "↑ (14.3%)"}
            ],
            "BTLS": [
                {"version": "25.1", "value": 282, "status": "MEDIUM RISK", "trend": "→"},
                {"version": "25.2", "value": 53, "status": "MEDIUM RISK", "trend": "↓ (81.2%)"},
                {"version": "25.3", "value": 29, "status": "ON TRACK", "trend": "↓ (45.3%)"}
            ]
        },
        "Open Security Defects": {
            "ATLS": [
                {"version": "25.1", "value": 9, "status": "MEDIUM RISK", "trend": "→"},
                {"version": "25.2", "value": 4, "status": "MEDIUM RISK", "trend": "↓ (55.6%)"},
                {"version": "25.3", "value": 43, "status": "RISK", "trend": "↑ (975.0%)"}
            ],
            "BTLS": [
                {"version": "25.1", "value": 21, "status": "MEDIUM RISK", "trend": "→"},
                {"version": "25.2", "value": 3, "status": "MEDIUM RISK", "trend": "↓ (85.7%)"},
                {"version": "25.3", "value": 0, "status": "ON TRACK", "trend": "↓ (100.0%)"}
            ]
        },
        "All Open Defects (T-1)": {
            "ATLS": [
                {"version": "25.1", "value": 339, "status": "RISK", "trend": "→"},
                {"version": "25.2", "value": 355, "status": "RISK", "trend": "↑ (4.7%)"},
                {"version": "25.3", "value": 382, "status": "RISK", "trend": "↑ (7.6%)"}
            ],
            "BTLS": [
                {"version": "25.1", "value": 715, "status": "RISK", "trend": "→"},
                {"version": "25.2", "value": 681, "status": "RISK", "trend": "↓ (4.8%)"},
                {"version": "25.3", "value": 689, "status": "RISK", "trend": "↑ (1.2%)"}
            ]
        },
        "All Security Open Defects": {
            "ATLS": [
                {"version": "25.1", "value": 66, "status": "RISK", "trend": "→"},
                {"version": "25.2", "value": 65, "status": "RISK", "trend": "↓ (1.5%)"},
                {"version": "25.3", "value": 103, "status": "RISK", "trend": "↑ (58.5%)"}
            ],
            "BTLS": [
                {"version": "25.1", "value": 103, "status": "RISK", "trend": "→"},
                {"version": "25.2", "value": 109, "status": "RISK", "trend": "↑ (5.8%)"},
                {"version": "25.3", "value": 70, "status": "RISK", "trend": "↓ (35.8%)"}
            ]
        },
        "Load/Performance": {
            "ATLS": [
                {"version": "25.1", "value": 10, "status": "MEDIUM RISK", "trend": "→"},
                {"version": "25.2", "value": 3, "status": "MEDIUM RISK", "trend": "↓ (70.0%)"},
                {"version": "25.3", "value": 2, "status": "RISK", "trend": "↓ (33.3%)"}
            ],
            "BTLS": [
                {"version": "25.1", "value": 11, "status": "MEDIUM RISK", "trend": "→"},
                {"version": "25.2", "value": 1, "status": "MEDIUM RISK", "trend": "↓ (90.9%)"},
                {"version": "25.3", "value": 1, "status": "RISK", "trend": "→"}
            ]
        },
        "E2E Test Coverage": [
            {"version": "25.1", "value": 1114, "status": "ON TRACK", "trend": "→"},
            {"version": "25.2", "value": 681, "status": "ON TRACK", "trend": "↓ (38.9%)"},
            {"version": "25.3", "value": 449, "status": "ON TRACK", "trend": "↓ (34.1%)"}
        ],
        "Automation Test Coverage": [
            {"version": "25.1", "value": 922, "status": "ON TRACK", "trend": "→"},
            {"version": "25.2", "value": 991, "status": "ON TRACK", "trend": "↑ (7.5%)"},
            {"version": "25.3", "value": 1042, "status": "ON TRACK", "trend": "↑ (5.1%)"}
        ],
        "Unit Test Coverage": [
            {"version": "25.1", "value": 5, "status": "WIP", "trend": "→"},
            {"version": "25.2", "value": 5, "status": "WIP", "trend": "→"},
            {"version": "25.3", "value": 5, "status": "WIP", "trend": "→"}
        ],
        "Defect Closure Rate": [
            {"version": "25.1", "value": 53.2, "status": "MEDIUM RISK", "trend": "→"},
            {"version": "25.2", "value": 65, "status": "MEDIUM RISK", "trend": "↑ (22.2%)"},
            {"version": "25.3", "value": 60.4, "status": "ON TRACK", "trend": "↓ (7.1%)"}
        ],
        "Regression Issues": [
            {"version": "25.1", "value": 2, "status": "MEDIUM RISK", "trend": "→"},
            {"version": "25.2", "value": 2, "status": "MEDIUM RISK", "trend": "→"},
            {"version": "25.3", "value": 0, "status": "ON TRACK", "trend": "↓ (100.0%)"}
        ]
    }
}

# Helper function to generate grouped bar charts
def generate_grouped_bar_chart(metric_name, atls_data, btls_data, filename):
    try:
        versions = [item['version'] for item in atls_data]
        atls_values = [item['value'] for item in atls_data]
        btls_values = [item['value'] for item in btls_data]

        x = range(len(versions))
        plt.figure(figsize=(8, 5), dpi=120)
        plt.bar(x, atls_values, width=0.4, label='ATLS', color='blue', align='center')
        plt.bar([p + 0.4 for p in x], btls_values, width=0.4, label='BTLS', color='orange', align='center')
        plt.xticks([p + 0.2 for p in x], versions)
        plt.xlabel('Version')
        plt.ylabel('Value')
        plt.title(metric_name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        logging.info(f"Generated chart: {filename}")
    except Exception as e:
        logging.error(f"Failed to generate chart for {metric_name}: {e}")

# Helper function to generate line charts
def generate_line_chart(metric_name, data, filename):
    try:
        versions = [item['version'] for item in data]
        values = [item['value'] for item in data]

        plt.figure(figsize=(8, 5), dpi=120)
        plt.plot(versions, values, marker='o', label=metric_name, color='green')
        plt.xlabel('Version')
        plt.ylabel('Value')
        plt.title(metric_name)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        logging.info(f"Generated chart: {filename}")
    except Exception as e:
        logging.error(f"Failed to generate chart for {metric_name}: {e}")

# Generate charts for ATLS and BTLS metrics
atls_btls_metrics = ['Open ALL RRR Defects', 'Open Security Defects', 'All Open Defects (T-1)', 'All Security Open Defects', 'Load/Performance']
for metric in atls_btls_metrics:
    if metric in data['metrics']:
        atls_data = data['metrics'][metric].get('ATLS', [])
        btls_data = data['metrics'][metric].get('BTLS', [])
        filename = os.path.join(output_dir, f"{metric.lower().replace(' ', '_')}_atls_btls.png")
        generate_grouped_bar_chart(metric, atls_data, btls_data, filename)

# Generate charts for coverage metrics
coverage_metrics = ['E2E Test Coverage', 'Automation Test Coverage', 'Unit Test Coverage']
for metric in coverage_metrics:
    if metric in data['metrics']:
        metric_data = data['metrics'][metric]
        filename = os.path.join(output_dir, f"{metric.lower().replace(' ', '_')}.png")
        generate_line_chart(metric, metric_data, filename)

# Generate charts for other metrics
other_metrics = ['Defect Closure Rate', 'Regression Issues']
for metric in other_metrics:
    if metric in data['metrics']:
        metric_data = data['metrics'][metric]
        filename = os.path.join(output_dir, f"{metric.lower().replace(' ', '_')}.png")
        generate_line_chart(metric, metric_data, filename)
