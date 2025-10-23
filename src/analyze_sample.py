"""
Sample Dataset Analysis

This script provides a quick overview of the sample dataset characteristics.
"""

def analyze_sample_dataset():
    """Analyze the sample dataset and display key statistics"""
    
    print("=" * 60)
    print("VENDOR PAYMENT SAMPLE DATASET ANALYSIS")
    print("=" * 60)
    
    import csv
    from collections import defaultdict
    from datetime import datetime
    
    # Read sample data
    with open('data/raw/vendor_payments_sample.csv', 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    
    print(f"\n📊 DATASET OVERVIEW")
    print("-" * 30)
    print(f"Total records: {len(rows)}")
    print(f"Features: {len(rows[0].keys())}")
    
    # Basic statistics
    delayed_count = sum(1 for row in rows if int(row['is_delayed']) == 1)
    delay_rate = delayed_count / len(rows)
    
    print(f"Delayed payments: {delayed_count} ({delay_rate:.1%})")
    print(f"On-time payments: {len(rows) - delayed_count} ({1-delay_rate:.1%})")
    
    # Date range
    dates = [row['invoice_date'] for row in rows]
    print(f"Date range: {min(dates)} to {max(dates)}")
    
    # Amount statistics
    amounts = [float(row['invoice_amount']) for row in rows]
    print(f"Amount range: ${min(amounts):,.2f} to ${max(amounts):,.2f}")
    print(f"Average amount: ${sum(amounts)/len(amounts):,.2f}")
    
    # Unique counts
    print(f"Unique vendors: {len(set(row['vendor_id'] for row in rows))}")
    print(f"Vendor categories: {len(set(row['vendor_category'] for row in rows))}")
    
    print(f"\n🏢 VENDOR CATEGORY BREAKDOWN")
    print("-" * 30)
    
    # Category analysis
    category_stats = defaultdict(lambda: {'total': 0, 'delayed': 0, 'amounts': []})
    
    for row in rows:
        category = row['vendor_category']
        amount = float(row['invoice_amount'])
        is_delayed = int(row['is_delayed'])
        
        category_stats[category]['total'] += 1
        category_stats[category]['amounts'].append(amount)
        if is_delayed:
            category_stats[category]['delayed'] += 1
    
    # Sort by delay rate
    sorted_categories = sorted(category_stats.items(), 
                             key=lambda x: x[1]['delayed']/x[1]['total'] if x[1]['total'] > 0 else 0, 
                             reverse=True)
    
    for category, stats in sorted_categories:
        delay_rate = stats['delayed'] / stats['total'] if stats['total'] > 0 else 0
        avg_amount = sum(stats['amounts']) / len(stats['amounts'])
        print(f"  {category:15}: {stats['total']:3d} records, {delay_rate:5.1%} delay rate, avg ${avg_amount:8,.0f}")
    
    print(f"\n💰 FINANCIAL ANALYSIS")
    print("-" * 30)
    
    # Financial impact
    total_amount = sum(amounts)
    delayed_amounts = [float(row['invoice_amount']) for row in rows if int(row['is_delayed']) == 1]
    amount_at_risk = sum(delayed_amounts)
    
    print(f"Total invoice value: ${total_amount:,.2f}")
    print(f"Amount at risk (delayed): ${amount_at_risk:,.2f}")
    print(f"Risk percentage: {amount_at_risk/total_amount:.1%}")
    
    # Payment terms analysis
    print(f"\n📅 PAYMENT TERMS ANALYSIS")
    print("-" * 30)
    
    terms_stats = defaultdict(lambda: {'total': 0, 'delayed': 0})
    for row in rows:
        terms = int(row['payment_terms'])
        is_delayed = int(row['is_delayed'])
        
        terms_stats[terms]['total'] += 1
        if is_delayed:
            terms_stats[terms]['delayed'] += 1
    
    for terms in sorted(terms_stats.keys()):
        stats = terms_stats[terms]
        delay_rate = stats['delayed'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {terms:2d} days: {stats['total']:3d} invoices, {delay_rate:5.1%} delay rate")
    
    print(f"\n🔍 KEY INSIGHTS FROM SAMPLE")
    print("-" * 30)
    
    # Find highest risk category
    highest_risk_cat = sorted_categories[0]
    print(f"• Highest risk category: {highest_risk_cat[0]} ({highest_risk_cat[1]['delayed']/highest_risk_cat[1]['total']:.1%} delay rate)")
    
    # Find largest amounts
    large_invoices = [row for row in rows if float(row['invoice_amount']) > 100000]
    if large_invoices:
        large_delay_rate = sum(1 for row in large_invoices if int(row['is_delayed']) == 1) / len(large_invoices)
        print(f"• Large invoices (>$100K): {len(large_invoices)} invoices, {large_delay_rate:.1%} delay rate")
    
    # Vendor with most past delays
    max_past_delays = max(int(row['vendor_past_delays']) for row in rows)
    high_risk_vendors = [row for row in rows if int(row['vendor_past_delays']) >= 5]
    if high_risk_vendors:
        high_risk_delay_rate = sum(1 for row in high_risk_vendors if int(row['is_delayed']) == 1) / len(high_risk_vendors)
        print(f"• High past delays (≥5): {len(high_risk_vendors)} invoices, {high_risk_delay_rate:.1%} delay rate")
    
    # Disputes impact
    dispute_invoices = [row for row in rows if int(row['has_dispute']) == 1]
    if dispute_invoices:
        dispute_delay_rate = sum(1 for row in dispute_invoices if int(row['is_delayed']) == 1) / len(dispute_invoices)
        print(f"• Invoices with disputes: {len(dispute_invoices)} invoices, {dispute_delay_rate:.1%} delay rate")
    
    print(f"\n✅ Sample dataset is ready for analysis and modeling!")
    print(f"📁 Location: data/raw/vendor_payments_sample.csv")
    print("=" * 60)

if __name__ == "__main__":
    analyze_sample_dataset()
