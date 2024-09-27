import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import PageBreak
from io import BytesIO
import pandas as pd
from evaluate.visualisation import combine_dicts
from evaluate.visualisation import table_vis
from evaluate.visualisation import attribute_vis

# Save the Matplotlib figure to an image in memory
def save_figure_to_image(fig):
    img_data = BytesIO()
    fig.savefig(img_data, format='PNG')
    plt.close(fig)
    img_data.seek(0)
    return img_data

def create_metric_table(privacy_scores, quality_scores, utility_scores):
    privacy_scores = {key: value for key, value in privacy_scores.items() if 'Total' in key}
    privacy_df = pd.DataFrame({'Privacy Metrics': privacy_scores.keys(), 
                                'Score': privacy_scores.values()})
    privacy_df['Privacy Metrics'] = privacy_df['Privacy Metrics'].str.replace(r'\bTotal\b', '', regex=True).str.strip()
    quality_scores = {key: value for key, value in quality_scores.items() if 'Total' in key}
    quality_df = pd.DataFrame({'Quality Metrics': quality_scores.keys(), 
                                'Score': quality_scores.values()})
    quality_df['Quality Metrics'] = quality_df['Quality Metrics'].str.replace(r'\bTotal\b', '', regex=True).str.strip()
    utility_scores = {key: value for key, value in utility_scores.items() if 'Total' in key}
    utility_df = pd.DataFrame({'Utility Metrics': utility_scores.keys(), 
                                'Score': utility_scores.values()})
    utility_df['Utility Metrics'] = utility_df['Utility Metrics'].str.replace(r'\bTotal\b', '', regex=True).str.strip()
    
    df = pd.concat([privacy_df, quality_df, utility_df], axis=1)

    return df

# Create the PDF report with text, a table, and a plot
def create_pdf_report(privacy_scores, quality_scores, utility_scores, data_columns):
    pdf_file = "Evaluation Report.pdf"
    pdf = SimpleDocTemplate(pdf_file, pagesize=A4)

    styles = getSampleStyleSheet()
    content = []
    subtitle_style = ParagraphStyle(name='Subtitle', fontSize=14, spaceAfter=10, textColor=colors.blue, alignment=TA_CENTER, fontName='Helvetica')
    subtitle_style2 = ParagraphStyle(name='Subtitle', fontSize=10, spaceAfter=10, textColor=colors.black, fontName='Helvetica-Bold')

    content.append(Paragraph("Synthetic Data Evaluation Report", styles['Title']))
    content.append(Paragraph("This report details the quality, privacy and utility evaluation metrics gained from the synthetic data, and visualisations to help interpret them. \n", styles['Normal']))
    content.append(Paragraph("<br/><br/>", styles['Normal']))
    content.append(Paragraph("Metrics Summary", subtitle_style))

    df = create_metric_table(privacy_scores, quality_scores, utility_scores)
    table_data = [df.columns.tolist()] + df.values.tolist()
    table = Table(table_data, hAlign='CENTER')
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

    fig = table_vis(privacy_scores, quality_scores, utility_scores)
    img_data = save_figure_to_image(fig)
    img = Image(img_data, width=440, height=275)  #  Specify the width and height of the image in the PDF
    content.append(img)

    content.append(Paragraph("<br/><br/>", styles['Normal']))
    level = 'Correlated Synthetic Data'
    content.append(Paragraph(f"Synthetic Data Categorisation Level: {level}", subtitle_style2))
    content.append(Paragraph("Correlated Synthetic Data is categorised as the highest risk due to capturing information about relationships and patterns between variables. Therefore, the privacy metrics should be evaluated carefully to ensure individuals arent at risk of being identifiable.", styles['Normal']))

    content.append(PageBreak())

    #### Boundary Adherence

    content.append(Paragraph("(Quality) Boundary Adherence Scores", subtitle_style))
    content.append(Paragraph("Boundary adherence measures whether values stay within the original min/max ranges of the data. (0.0: means none of the attributes have the same min/max ranges, 1.0: means all attributes have the same min/max ranges)", styles['Normal']))

    content.append(Paragraph("<br/><br/>", styles['Normal']))
    fig = attribute_vis("Boundary Adherence Individual", quality_scores, data_columns)
    img_data = save_figure_to_image(fig)
    img = Image(img_data, width=504, height=216)
    content.append(img)

    #### Coverage

    content.append(Paragraph("<br/><br/>", styles['Normal']))

    content.append(Paragraph("(Quality) Coverage Scores", subtitle_style))
    content.append(Paragraph("Coverage measures whether the whole range of values are represented. (0.0: means none of the values are represented, 1.0: means all values are represented)", styles['Normal']))

    content.append(Paragraph("<br/><br/>", styles['Normal']))
    fig = attribute_vis("Coverage Individual", quality_scores, data_columns)
    img_data = save_figure_to_image(fig)
    img = Image(img_data, width=504, height=216)
    content.append(img)

    content.append(PageBreak())

    #### Complement

    content.append(Paragraph("<br/><br/>", styles['Normal']))

    content.append(Paragraph("(Quality) Complement Scores", subtitle_style))
    content.append(Paragraph("Complement measures whether the distributions look the same. (0.0: means the distributions are as different as they can be, 1.0: means the distributions are exactly the same)", styles['Normal']))

    content.append(Paragraph("<br/><br/>", styles['Normal']))
    fig = attribute_vis("Complement Individual", quality_scores, data_columns)
    img_data = save_figure_to_image(fig)
    img = Image(img_data, width=504, height=216)
    content.append(img)
    

    pdf.build(content)
    print(f"PDF report created: {pdf_file}")
