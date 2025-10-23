"""
ER Model Generator for Vendor Payment Delay Prediction System
This script generates an Entity-Relationship diagram showing the database structure
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_er_diagram():
    """Create and display the ER diagram for the vendor payment system"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    entity_color = '#E8F4FD'
    attribute_color = '#FFF2CC'
    relationship_color = '#F8CECC'
    
    # Entity: Vendor
    vendor_box = FancyBboxPatch((1, 8), 3, 2.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=entity_color, 
                               edgecolor='black', 
                               linewidth=2)
    ax.add_patch(vendor_box)
    ax.text(2.5, 9.7, 'VENDOR', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(2.5, 9.3, 'vendor_id (PK)', ha='center', va='center', fontsize=9)
    ax.text(2.5, 9.0, 'vendor_category', ha='center', va='center', fontsize=9)
    ax.text(2.5, 8.7, 'vendor_past_delays', ha='center', va='center', fontsize=9)
    ax.text(2.5, 8.4, 'vendor_relationship_years', ha='center', va='center', fontsize=9)
    ax.text(2.5, 8.1, 'payment_frequency', ha='center', va='center', fontsize=9)
    
    # Entity: Invoice
    invoice_box = FancyBboxPatch((7, 8), 3, 2.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=entity_color, 
                                edgecolor='black', 
                                linewidth=2)
    ax.add_patch(invoice_box)
    ax.text(8.5, 9.7, 'INVOICE', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(8.5, 9.3, 'invoice_id (PK)', ha='center', va='center', fontsize=9)
    ax.text(8.5, 9.0, 'vendor_id (FK)', ha='center', va='center', fontsize=9)
    ax.text(8.5, 8.7, 'invoice_date', ha='center', va='center', fontsize=9)
    ax.text(8.5, 8.4, 'due_date', ha='center', va='center', fontsize=9)
    ax.text(8.5, 8.1, 'invoice_amount', ha='center', va='center', fontsize=9)
    
    # Entity: Payment
    payment_box = FancyBboxPatch((12, 8), 3, 2.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=entity_color, 
                                edgecolor='black', 
                                linewidth=2)
    ax.add_patch(payment_box)
    ax.text(13.5, 9.7, 'PAYMENT', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(13.5, 9.3, 'payment_id (PK)', ha='center', va='center', fontsize=9)
    ax.text(13.5, 9.0, 'invoice_id (FK)', ha='center', va='center', fontsize=9)
    ax.text(13.5, 8.7, 'actual_payment_date', ha='center', va='center', fontsize=9)
    ax.text(13.5, 8.4, 'days_difference', ha='center', va='center', fontsize=9)
    ax.text(13.5, 8.1, 'is_delayed', ha='center', va='center', fontsize=9)
    
    # Entity: Payment Terms
    terms_box = FancyBboxPatch((4, 5), 3, 2, 
                              boxstyle="round,pad=0.1", 
                              facecolor=entity_color, 
                              edgecolor='black', 
                              linewidth=2)
    ax.add_patch(terms_box)
    ax.text(5.5, 6.5, 'PAYMENT_TERMS', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5.5, 6.1, 'terms_id (PK)', ha='center', va='center', fontsize=9)
    ax.text(5.5, 5.8, 'payment_terms', ha='center', va='center', fontsize=9)
    ax.text(5.5, 5.5, 'quarter', ha='center', va='center', fontsize=9)
    ax.text(5.5, 5.2, 'year', ha='center', va='center', fontsize=9)
    
    # Entity: Cash Flow
    cashflow_box = FancyBboxPatch((10, 5), 3, 2, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=entity_color, 
                                 edgecolor='black', 
                                 linewidth=2)
    ax.add_patch(cashflow_box)
    ax.text(11.5, 6.5, 'CASH_FLOW', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(11.5, 6.1, 'cashflow_id (PK)', ha='center', va='center', fontsize=9)
    ax.text(11.5, 5.8, 'cash_flow_level', ha='center', va='center', fontsize=9)
    ax.text(11.5, 5.5, 'month_cash_available', ha='center', va='center', fontsize=9)
    ax.text(11.5, 5.2, 'economic_indicator', ha='center', va='center', fontsize=9)
    
    # Entity: Dispute
    dispute_box = FancyBboxPatch((1, 2), 3, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=entity_color, 
                                edgecolor='black', 
                                linewidth=2)
    ax.add_patch(dispute_box)
    ax.text(2.5, 3, 'DISPUTE', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(2.5, 2.6, 'dispute_id (PK)', ha='center', va='center', fontsize=9)
    ax.text(2.5, 2.3, 'invoice_id (FK)', ha='center', va='center', fontsize=9)
    ax.text(2.5, 2.0, 'has_dispute', ha='center', va='center', fontsize=9)
    
    # Relationships (Diamond shapes)
    # Has relationship between Vendor and Invoice
    has_rel = patches.Polygon([(5.5, 9.25), (6, 8.75), (5.5, 8.25), (5, 8.75)], 
                             facecolor=relationship_color, edgecolor='black')
    ax.add_patch(has_rel)
    ax.text(5.5, 8.75, 'HAS', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Results in relationship between Invoice and Payment
    results_rel = patches.Polygon([(10.5, 9.25), (11, 8.75), (10.5, 8.25), (10, 8.75)], 
                                 facecolor=relationship_color, edgecolor='black')
    ax.add_patch(results_rel)
    ax.text(10.5, 8.75, 'RESULTS\nIN', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Governed by relationship between Invoice and Payment Terms
    governed_rel = patches.Polygon([(6.5, 7), (7, 6.5), (6.5, 6), (6, 6.5)], 
                                  facecolor=relationship_color, edgecolor='black')
    ax.add_patch(governed_rel)
    ax.text(6.5, 6.5, 'GOVERNED\nBY', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Influenced by relationship between Payment and Cash Flow
    influenced_rel = patches.Polygon([(11.5, 7), (12, 6.5), (11.5, 6), (11, 6.5)], 
                                    facecolor=relationship_color, edgecolor='black')
    ax.add_patch(influenced_rel)
    ax.text(11.5, 6.5, 'INFLUENCED\nBY', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # May have relationship between Invoice and Dispute
    may_have_rel = patches.Polygon([(5.5, 4.5), (6, 4), (5.5, 3.5), (5, 4)], 
                                  facecolor=relationship_color, edgecolor='black')
    ax.add_patch(may_have_rel)
    ax.text(5.5, 4, 'MAY\nHAVE', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Connection lines
    # Vendor to Has relationship
    ax.plot([4, 5], [9.25, 8.75], 'k-', linewidth=2)
    # Has relationship to Invoice
    ax.plot([6, 7], [8.75, 9.25], 'k-', linewidth=2)
    
    # Invoice to Results relationship
    ax.plot([10, 10], [9.25, 8.75], 'k-', linewidth=2)
    # Results relationship to Payment
    ax.plot([11, 12], [8.75, 9.25], 'k-', linewidth=2)
    
    # Invoice to Governed relationship
    ax.plot([8.5, 6.5], [8, 6.5], 'k-', linewidth=2)
    # Governed relationship to Payment Terms
    ax.plot([6.5, 5.5], [6.5, 6.5], 'k-', linewidth=2)
    
    # Payment to Influenced relationship
    ax.plot([13.5, 11.5], [8, 6.5], 'k-', linewidth=2)
    # Influenced relationship to Cash Flow
    ax.plot([11.5, 11.5], [6.5, 6.5], 'k-', linewidth=2)
    
    # Invoice to May Have relationship
    ax.plot([8.5, 5.5], [8, 4], 'k-', linewidth=2)
    # May Have relationship to Dispute
    ax.plot([5.5, 2.5], [4, 3.5], 'k-', linewidth=2)
    
    # Cardinality labels
    ax.text(4.5, 9.5, '1', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(6.5, 9.5, 'M', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(10.5, 9.5, '1', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(11.5, 9.5, '1', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(7.5, 7, 'M', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5.5, 7, '1', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(12.5, 7, 'M', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(11.5, 7, '1', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(7, 4.5, '1', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(3.5, 3.5, '1', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Title
    ax.text(8, 11.5, 'Vendor Payment Delay Prediction System - ER Model', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Legend
    legend_y = 1
    ax.add_patch(FancyBboxPatch((0.5, legend_y-0.5), 1, 0.3, 
                               boxstyle="round,pad=0.05", 
                               facecolor=entity_color, 
                               edgecolor='black'))
    ax.text(1, legend_y-0.35, 'Entity', ha='center', va='center', fontsize=9)
    
    ax.add_patch(patches.Polygon([(2.5, legend_y-0.25), (2.75, legend_y-0.45), 
                                 (2.5, legend_y-0.65), (2.25, legend_y-0.45)], 
                                facecolor=relationship_color, edgecolor='black'))
    ax.text(2.5, legend_y-0.45, 'Relationship', ha='center', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save the diagram
    output_path = '/Users/sreevanthsv/Desktop/DWDM/reports/figures/er_model.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ER Model saved to: {output_path}")
    
    plt.show()

def create_detailed_schema():
    """Create a detailed database schema diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Table definitions
    tables = {
        'vendors': {
            'pos': (1, 7),
            'size': (3, 2.5),
            'fields': [
                'vendor_id (PK) VARCHAR(10)',
                'vendor_category VARCHAR(50)',
                'vendor_past_delays INT',
                'vendor_relationship_years DECIMAL(3,2)',
                'payment_frequency INT',
                'created_date DATETIME',
                'updated_date DATETIME'
            ]
        },
        'invoices': {
            'pos': (5.5, 7),
            'size': (3, 2.5),
            'fields': [
                'invoice_id (PK) VARCHAR(15)',
                'vendor_id (FK) VARCHAR(10)',
                'invoice_date DATE',
                'due_date DATE',
                'invoice_amount DECIMAL(10,2)',
                'payment_terms INT',
                'quarter VARCHAR(2)',
                'year INT',
                'has_dispute BOOLEAN'
            ]
        },
        'payments': {
            'pos': (10, 7),
            'size': (3, 2.5),
            'fields': [
                'payment_id (PK) VARCHAR(15)',
                'invoice_id (FK) VARCHAR(15)',
                'actual_payment_date DATE',
                'days_difference INT',
                'is_delayed BOOLEAN',
                'payment_method VARCHAR(20)',
                'processed_by VARCHAR(50)'
            ]
        },
        'cash_flow': {
            'pos': (3, 3.5),
            'size': (3, 2),
            'fields': [
                'cashflow_id (PK) INT',
                'period_date DATE',
                'cash_flow_level VARCHAR(10)',
                'month_cash_available DECIMAL(12,2)',
                'economic_indicator DECIMAL(5,2)',
                'forecast_accuracy DECIMAL(3,2)'
            ]
        },
        'disputes': {
            'pos': (7.5, 3.5),
            'size': (3, 2),
            'fields': [
                'dispute_id (PK) INT',
                'invoice_id (FK) VARCHAR(15)',
                'dispute_date DATE',
                'dispute_reason VARCHAR(100)',
                'resolution_date DATE',
                'status VARCHAR(20)'
            ]
        }
    }
    
    # Draw tables
    for table_name, table_info in tables.items():
        x, y = table_info['pos']
        w, h = table_info['size']
        
        # Table box
        table_box = FancyBboxPatch((x, y), w, h, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#E8F4FD', 
                                  edgecolor='black', 
                                  linewidth=2)
        ax.add_patch(table_box)
        
        # Table name
        ax.text(x + w/2, y + h - 0.2, table_name.upper(), 
                ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Table fields
        field_y = y + h - 0.5
        for field in table_info['fields']:
            ax.text(x + 0.1, field_y, field, ha='left', va='center', fontsize=8)
            field_y -= 0.25
    
    # Draw relationships
    # Vendor to Invoice
    ax.annotate('', xy=(5.5, 8.25), xytext=(4, 8.25),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.text(4.75, 8.5, '1:M', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Invoice to Payment
    ax.annotate('', xy=(10, 8.25), xytext=(8.5, 8.25),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.text(9.25, 8.5, '1:1', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Invoice to Dispute
    ax.annotate('', xy=(7.5, 4.5), xytext=(7, 7),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.text(7.5, 5.75, '1:0..1', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Title
    ax.text(7, 9.5, 'Database Schema - Vendor Payment System', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the schema
    output_path = '/Users/sreevanthsv/Desktop/DWDM/reports/figures/database_schema.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Database Schema saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    import os
    
    # Create output directory if it doesn't exist
    output_dir = '/Users/sreevanthsv/Desktop/DWDM/reports/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating ER Model and Database Schema...")
    
    # Generate ER Model
    create_er_diagram()
    
    # Generate Database Schema
    create_detailed_schema()
    
    print("ER Model and Database Schema generated successfully!")