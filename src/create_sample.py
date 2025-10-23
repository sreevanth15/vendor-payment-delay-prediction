"""
Create Sample Dataset - 200 records from vendor payments data

This script creates a representative sample of 200 records from the full dataset
while maintaining the same class distribution and key characteristics.
"""

import csv
import random

def create_sample_dataset():
    """Create a sample dataset of 200 records"""
    
    print("Creating sample dataset from vendor payments data...")
    
    # Read the original dataset
    with open('data/raw/vendor_payments.csv', 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Get header row
        all_rows = list(reader)
    
    print(f"Original dataset: {len(all_rows)} records")
    
    # Separate delayed and on-time payments
    delayed_rows = []
    ontime_rows = []
    
    for row in all_rows:
        is_delayed = int(row[-1])  # Last column is is_delayed
        if is_delayed == 1:
            delayed_rows.append(row)
        else:
            ontime_rows.append(row)
    
    print(f"Delayed payments: {len(delayed_rows)} ({len(delayed_rows)/len(all_rows):.1%})")
    print(f"On-time payments: {len(ontime_rows)} ({len(ontime_rows)/len(all_rows):.1%})")
    
    # Calculate proportional sample sizes
    total_sample = 200
    delayed_sample_size = int(total_sample * len(delayed_rows) / len(all_rows))
    ontime_sample_size = total_sample - delayed_sample_size
    
    print(f"Sample delayed payments: {delayed_sample_size}")
    print(f"Sample on-time payments: {ontime_sample_size}")
    
    # Random seed for reproducibility
    random.seed(42)
    
    # Sample records
    sample_delayed = random.sample(delayed_rows, min(delayed_sample_size, len(delayed_rows)))
    sample_ontime = random.sample(ontime_rows, min(ontime_sample_size, len(ontime_rows)))
    
    # Combine samples
    sample_data = sample_delayed + sample_ontime
    
    # Shuffle the combined sample
    random.shuffle(sample_data)
    
    # Sort by invoice_id for better organization
    sample_data.sort(key=lambda x: x[1])  # Sort by invoice_id (column 1)
    
    # Write sample dataset
    with open('data/raw/vendor_payments_sample.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(sample_data)
    
    print(f"Sample dataset created: {len(sample_data)} records")
    print(f"Saved to: data/raw/vendor_payments_sample.csv")
    
    # Calculate sample statistics
    sample_delayed_count = sum(1 for row in sample_data if int(row[-1]) == 1)
    sample_delay_rate = sample_delayed_count / len(sample_data)
    
    print(f"Sample delay rate: {sample_delay_rate:.1%}")
    
    # Show vendor category distribution
    categories = {}
    for row in sample_data:
        category = row[2]  # vendor_category is column 2
        if category not in categories:
            categories[category] = {'total': 0, 'delayed': 0}
        categories[category]['total'] += 1
        if int(row[-1]) == 1:
            categories[category]['delayed'] += 1
    
    print(f"\nVendor Category Distribution in Sample:")
    for category, stats in categories.items():
        delay_rate = stats['delayed'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {category}: {stats['total']} records ({delay_rate:.1%} delay rate)")
    
    # Show amount statistics
    amounts = [float(row[6]) for row in sample_data]  # invoice_amount is column 6
    print(f"\nAmount Statistics:")
    print(f"  Min: ${min(amounts):,.2f}")
    print(f"  Max: ${max(amounts):,.2f}")
    print(f"  Average: ${sum(amounts)/len(amounts):,.2f}")
    
    return len(sample_data)

if __name__ == "__main__":
    sample_count = create_sample_dataset()
    print(f"\n✅ Sample dataset creation completed!")
    print(f"📁 File: data/raw/vendor_payments_sample.csv")
    print(f"📊 Records: {sample_count}")
