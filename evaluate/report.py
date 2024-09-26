import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import pandas as pd
from evaluate.visualisation import combine_dicts

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

def create_metric_table(privacy_scores, quality_scores, utility_scores):
    combined = combine_dicts(privacy_scores, quality_scores, utility_scores)
    total_combined = {key: value for key, value in combined.items() if 'Total' in key}
    x = list(total_combined.keys())
    y = list(total_combined.values())
    data = {'Metric Name': x, 'Score': y}
    df = pd.DataFrame(data)
    return df

# Create the PDF report with text, a table, and a plot
def create_pdf_report(privacy_scores, quality_scores, utility_scores, data_columns):
    pdf_file = "Evaluation Report.pdf"
    pdf = SimpleDocTemplate(pdf_file, pagesize=A4)

    styles = getSampleStyleSheet()
    content = []
    subtitle_style = ParagraphStyle(name='Subtitle', fontSize=14, spaceAfter=10, textColor=colors.blue)

    content.append(Paragraph("Synthetic Data Evaluation Report", styles['Title']))
    content.append(Paragraph("This report details the quality, privacy and utility evaluation metrics gained from the synthetic data, and visualisations to help interpret them. \n", styles['Normal']))
    content.append(Paragraph("<br/><br/>", styles['Normal']))
    content.append(Paragraph("Metrics Summary", subtitle_style))

    df = create_metric_table(privacy_scores, quality_scores, utility_scores)
    table_data = [df.columns.tolist()] + df.values.tolist()
    table = Table(table_data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.white),
    ]))
    content.append(table)

    content.append(Paragraph("<br/><br/>", styles['Normal']))

    fig = create_matplotlib_figure()
    img_data = save_figure_to_image(fig)
    img = Image(img_data, width=400, height=300)  # Specify the width and height of the image in the PDF
    content.append(img)

    pdf.build(content)
    print(f"PDF report created: {pdf_file}")
