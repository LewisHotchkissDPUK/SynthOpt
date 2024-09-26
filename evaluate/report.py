import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import pandas as pd

# Function to create a Matplotlib figure
def create_matplotlib_figure():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label="Sample Plot")
    ax.set_title("Sample Matplotlib Plot")
    ax.set_xlabel("X-Axis")
    ax.set_ylabel("Y-Axis")
    ax.legend()
    return fig

# Save the Matplotlib figure to an image in memory
def save_figure_to_image(fig):
    img_data = BytesIO()
    fig.savefig(img_data, format='PNG')
    plt.close(fig)
    img_data.seek(0)
    return img_data

# Create a sample table using Pandas
def create_sample_table():
    data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
            'Age': [25, 30, 35, 40],
            'Occupation': ['Engineer', 'Doctor', 'Artist', 'Lawyer']}
    df = pd.DataFrame(data)
    return df

# Convert a Pandas DataFrame into a format suitable for ReportLab Table
def convert_df_to_table_data(df):
    # Convert DataFrame to list of lists
    table_data = [df.columns.tolist()] + df.values.tolist()
    return table_data

# Create the PDF report with text, a table, and a plot
def create_pdf_report():
    # Set up the PDF document with A4 page size
    pdf_file = "A4_report_with_plot_and_table.pdf"
    pdf = SimpleDocTemplate(pdf_file, pagesize=A4)

    # Get the sample style for text
    styles = getSampleStyleSheet()
    content = []

    # Add a title
    content.append(Paragraph("A4 Report with Text, Table, and Plot", styles['Title']))

    # Add some text
    content.append(Paragraph("This is a sample report that includes a table and a Matplotlib plot.", styles['Normal']))

    # Add a table
    df = create_sample_table()
    table_data = convert_df_to_table_data(df)
    
    # Create the Table object
    table = Table(table_data)

    # Style the table
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    # Add the table to the content
    content.append(table)

    # Add some space between table and plot
    content.append(Paragraph("<br/><br/>", styles['Normal']))

    # Add the Matplotlib plot as a flowable Image
    fig = create_matplotlib_figure()
    img_data = save_figure_to_image(fig)
    
    # Add the image to the flowable content
    img = Image(img_data, width=400, height=300)  # Specify the width and height of the image in the PDF
    content.append(img)

    # Build the PDF
    pdf.build(content)

    print(f"PDF report created: {pdf_file}")

# Generate the PDF report
create_pdf_report()
