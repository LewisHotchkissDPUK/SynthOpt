import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import PageBreak
from reportlab.lib.colors import HexColor
from io import BytesIO
import pandas as pd
from synthopt.evaluate.visualisation import combine_dicts
from synthopt.evaluate.visualisation import table_vis
from synthopt.evaluate.visualisation import attribute_vis
from synthopt.evaluate.visualisation import distribution_vis, correlation_vis
from synthopt.evaluate.visualisation import reduction_vis
from synthopt import evaluate
from functools import reduce
import os
from PIL import Image as PILImage
import requests
from io import StringIO 

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
def create_pdf_report(privacy_scores, quality_scores, utility_scores, table_type, identifier_column, data, synthetic_data, data_columns, save_location):
    if table_type == 'multi':
        data = reduce(lambda left, right: pd.merge(left, right, on=identifier_column), data)
        synthetic_data = reduce(lambda left, right: pd.merge(left, right, on=identifier_column), synthetic_data)
    data = data.drop(columns=[identifier_column])
    synthetic_data = synthetic_data.drop(columns=[identifier_column])

    pdf_file = save_location
    pdf = SimpleDocTemplate(pdf_file, pagesize=A4)

    styles = getSampleStyleSheet()
    content = []
    subtitle_style = ParagraphStyle(name='Subtitle', fontSize=14, spaceAfter=10, textColor=HexColor("#3f1af6"), alignment=TA_CENTER, fontName='Helvetica')
    subtitle_style2 = ParagraphStyle(name='Subtitle', fontSize=10, spaceAfter=10, textColor=colors.black, fontName='Helvetica-Bold')

    red_style = ParagraphStyle(
        name='RedNormal',
        parent=styles['Normal'],
        textColor=HexColor('#e92751'),  # Hex code for red
        fontName='Helvetica',
        fontSize=10
    )
    green_style = ParagraphStyle(
        name='GreenNormal',
        parent=styles['Normal'],
        textColor=HexColor('#29c96a'),  # Hex code for green
        fontName='Helvetica',
        fontSize=10
    )

    content.append(Paragraph("Synthetic Data Evaluation Report", styles['Title']))
    content.append(Paragraph("This report details the quality, privacy and utility evaluation metrics gained from the synthetic data, and visualisations to help interpret them. \n", styles['Normal']))
    content.append(Paragraph("<br/><br/>", styles['Normal']))
    content.append(Paragraph("Metrics Summary", subtitle_style))

    df = create_metric_table(privacy_scores, quality_scores, utility_scores)
    table_data = [df.columns.tolist()] + df.values.tolist()
    table = Table(table_data, hAlign='CENTER')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#3f1af6")), #colors.blue
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor("#ececec")), #colors.lightblue
        ('GRID', (0, 0), (-1, -1), 1, colors.white),
    ]))
    content.append(table)

    level = ""
    desc = ""

    if quality_scores['Boundary Adherence Total'] < 0.7 and quality_scores['Coverage Total'] < 0.7 and quality_scores['Complement Total'] < 0.7 and utility_scores['Statistic Similarity Total'] < 0.7 and utility_scores['Correlation Total'] < 0.7:
        level = "Random Data"
        desc = ""
    if quality_scores['Boundary Adherence Total'] >= 0.7 and quality_scores['Coverage Total'] >= 0.7 and quality_scores['Complement Total'] < 0.7 and utility_scores['Statistic Similarity Total'] < 0.7 and utility_scores['Correlation Total'] < 0.7:
        level = "Structural Synthetic Data"
        desc = "Structural synthetic data is classified as the lowest fidelity of synthetic data. This means that the data value ranges are the same, and a majority of the values are represented, but there is no statistial similarity. This means there is a very low privacy risk as no statistics or correlations are captured from the real data."
    if quality_scores['Boundary Adherence Total'] >= 0.7 and quality_scores['Coverage Total'] >= 0.7 and quality_scores['Complement Total'] >= 0.7 and utility_scores['Statistic Similarity Total'] >= 0.7 and utility_scores['Correlation Total'] < 0.7:
        level = "Statistical Synthetic Data"
        desc = "Statistical synthetic data means that the distributions and summary statistics are roughly the same compared to the real data but that no relationships between the features are captured. This means that there may be some privacy risk, but is much lower compared to correlated data."
    if quality_scores['Boundary Adherence Total'] >= 0.7 and quality_scores['Coverage Total'] >= 0.7 and quality_scores['Complement Total'] >= 0.7 and utility_scores['Statistic Similarity Total'] >= 0.7 and utility_scores['Correlation Total'] >= 0.7:
        level = "Correlated Synthetic Data"
        desc = "Correlated Synthetic Data is categorised as the highest risk due to capturing information about relationships and patterns between variables. Therefore, the privacy metrics should be evaluated carefully to ensure individuals arent at risk of being identifiable."

    content.append(Paragraph("<br/><br/>", styles['Normal']))
    content.append(Paragraph(f"Synthetic Data Categorisation Level: {level}", subtitle_style2))
    content.append(Paragraph(f"{desc}", styles['Normal']))

    content.append(Paragraph("<br/><br/>", styles['Normal']))

    try:
        response = requests.get("https://github.com/LewisHotchkissDPUK/SynthOpt/blob/46027d9ece6a65997876ef6bbb121ab6f27a58ab/synthopt/evaluate/sds.png")
        img = PILImage.open(StringIO(response.content))
        external_img = Image(img, width=436, height=260)  # Adjust width and height based on image size
        #image_path = os.path.join(os.path.dirname(__file__), '..', 'evaluate', 'sds.png')
        #image_path = os.path.join(evaluate.__path__[0], 'sds.png')
        #external_img = Image(image_path, width=436, height=260)  # Adjust width and height based on image size

        content.append(external_img)
    except Exception:
        print("couldnt load image")

    #### Boundary Adherence

    content.append(PageBreak())

    content.append(Paragraph("Boundary Adherence Scores", subtitle_style))
    content.append(Paragraph("Boundary adherence measures whether values stay within the original min/max ranges of the data. (0.0: means none of the attributes have the same min/max ranges, 1.0: means all attributes have the same min/max ranges)", styles['Normal']))

    content.append(Paragraph("<br/><br/>", styles['Normal']))
    fig = attribute_vis("Boundary Adherence Individual", quality_scores, data_columns)
    img_data = save_figure_to_image(fig)
    img = Image(img_data, width=504, height=216)
    content.append(img)

    #### Coverage

    content.append(Paragraph("<br/><br/>", styles['Normal']))

    content.append(Paragraph("Coverage Scores", subtitle_style))
    content.append(Paragraph("Coverage measures whether the whole range of values are represented. (0.0: means none of the values are represented, 1.0: means all values are represented)", styles['Normal']))

    content.append(Paragraph("<br/><br/>", styles['Normal']))
    fig = attribute_vis("Coverage Individual", quality_scores, data_columns)
    img_data = save_figure_to_image(fig)
    img = Image(img_data, width=504, height=216)
    content.append(img)

    #### Complement

    content.append(PageBreak())

    content.append(Paragraph("Complement Scores", subtitle_style))
    content.append(Paragraph("Complement measures whether the distributions look the same. (0.0: means the distributions are as different as they can be, 1.0: means the distributions are exactly the same)", styles['Normal']))

    content.append(Paragraph("<br/><br/>", styles['Normal']))
    fig = attribute_vis("Complement Individual", quality_scores, data_columns)
    img_data = save_figure_to_image(fig)
    img = Image(img_data, width=504, height=216)
    content.append(img)

    #### Similarity

    content.append(Paragraph("<br/><br/>", styles['Normal']))

    content.append(Paragraph("Similarity Scores", subtitle_style))
    content.append(Paragraph("Statistic similarity measures how similar the summary statistics are such as mean and standard deviation. (0.0: means the summary statistics are extremely different to each other, 1.0: means the summary statistics are exactly the same)", styles['Normal']))

    content.append(Paragraph("<br/><br/>", styles['Normal']))
    fig = attribute_vis("Statistic Similarity Individual", utility_scores, data_columns)
    img_data = save_figure_to_image(fig)
    img = Image(img_data, width=504, height=216)
    content.append(img)

    ####

    content.append(PageBreak())

    content.append(Paragraph("Example Distribution Comparisons", subtitle_style))

    content.append(Paragraph("<br/><br/>", styles['Normal']))
    fig = distribution_vis(data, synthetic_data, data_columns)
    img_data = save_figure_to_image(fig)
    img = Image(img_data, width=500, height=555)
    content.append(img)

    ####

    content.append(PageBreak())

    content.append(Paragraph("Example Correlation Comparisons", subtitle_style))

    content.append(Paragraph("<br/><br/>", styles['Normal']))
    fig = correlation_vis(data, synthetic_data, data_columns)
    img_data = save_figure_to_image(fig)
    img = Image(img_data, width=500, height=555)
    content.append(img)

    #for each set of scores maybe include a rating
    #show some distribution and correlation plot comparisons for real vs fake, make sure its smoothed with no points.
    
    content.append(PageBreak())
    content.append(Paragraph("Meaning of Metrics", subtitle_style))
    content.append(Paragraph("<br/><br/>", styles['Normal']))

    content.append(Paragraph(f"(Privacy) Exact Matches", subtitle_style2))
    content.append(Paragraph("This metric measures whether each row in the synthetic data is novel, or whether it exactly matches an original row in the real data.", styles['Normal']))
    content.append(Paragraph(f" ", subtitle_style2))
    content.append(Paragraph("(best) 1.0: The rows in the synthetic data are all new. There are no matches with the real data.", green_style))
    content.append(Paragraph("(worst) 0.0: All the rows in the synthetic data are copies of rows in the real data.", red_style))
    content.append(Paragraph("<br/><br/>", styles['Normal']))

    content.append(Paragraph(f"(Privacy) Detection", subtitle_style2))
    content.append(Paragraph("This metric calculate how difficult it is to tell apart the real data from the synthetic data using machine learning techniques.A score of 1 may indicate high quality but it could also be a clue that the synthetic data is leaking privacy (for example, if the synthetic data is copying the rows in the real data).", styles['Normal']))
    content.append(Paragraph(f" ", subtitle_style2))
    content.append(Paragraph("(worst) 1.0: The machine learning model cannot identify the synthetic data apart from the real data.", red_style))
    content.append(Paragraph("(best) 0.0: The machine learning model can perfectly identify synthetic data apart from the real data.", green_style))
    content.append(Paragraph("<br/><br/>", styles['Normal']))

    content.append(Paragraph(f"(Privacy) Inference Protection", subtitle_style2))
    content.append(Paragraph("This metric calculates the risk of an attacker being able to infer real, sensitive values. It is assumed that an attacker already possess a few columns of real data; they will combine it with the synthetic data to make educated guesses.", styles['Normal']))
    content.append(Paragraph(f" ", subtitle_style2))
    content.append(Paragraph("(best) 1.0: The real data is 100% safe from the attack. The attacker is not able to correctly guess any of the sensitive values by applying the chosen attack algorithm.", green_style))
    content.append(Paragraph("(worst) 0.0: The real data is not at all safe from the attack. The attacker is able to correctly guess every sensitive value by applying the chosen attack algorithm.", red_style))
    content.append(Paragraph("<br/><br/>", styles['Normal']))

    content.append(Paragraph(f"(Privacy) Singling Out Risk", subtitle_style2))
    content.append(Paragraph("This metric measures how much the synthetic data can help an attacker finding a combination of attributes that single out records in the training data. This attack evaluates the robustness of the synthetic data to finding unique values of some attribute which single out an individual.", styles['Normal']))
    content.append(Paragraph(f" ", subtitle_style2))
    content.append(Paragraph("(worst) 1.0: There is a high risk that an individual can be singled out by a unique combination of their attributes.", red_style))
    content.append(Paragraph("(best) 0.0: There is a low risk that an individual can be singled out by a unique combination of their attributes.", green_style))
    content.append(Paragraph("<br/><br/>", styles['Normal']))

    content.append(Paragraph(f"(Privacy) Linkability Risk", subtitle_style2))
    content.append(Paragraph("This metric measures how much the synthetic data will help an adversary who tries to link two other datasets based on a subset of attributes. For example, suppose that the adversary finds dataset A containing, among other fields, information about the profession and education of people, and dataset B containing some demographic and health related information. Can the attacker use the synthetic dataset to link these two datasets? (It is assumed the attacker knows the key fields of the individual and that they are split across different datasets.)", styles['Normal']))
    content.append(Paragraph(f" ", subtitle_style2))
    content.append(Paragraph("(worst) 1.0: There is a high risk that attributes can be linked to identify an individual.", red_style))
    content.append(Paragraph("(best) 0.0: There is a low risk that attributes can be linked to identify an individual.", green_style))
    content.append(Paragraph("<br/><br/>", styles['Normal']))

    content.append(Paragraph(f"(Privacy) Inference Risk", subtitle_style2))
    content.append(Paragraph("This metric measures the inference risk. It does so by measuring the success of an attacker that tries to discover the value of some secret attribute for a set of target records on which some auxiliary knowledge is available. (Again, like the linkability risk, it assumes the attacker knows the key fields)", styles['Normal']))
    content.append(Paragraph(f" ", subtitle_style2))
    content.append(Paragraph("(worst) 1.0: There is a high risk that sensitive attributes can be inferred.", red_style))
    content.append(Paragraph("(best) 0.0: There is a low risk that sensitive attributes can be inferred.", green_style))
    content.append(Paragraph("<br/><br/>", styles['Normal']))

    content.append(Paragraph(f"(Quality) Boundary Adherence", subtitle_style2))
    content.append(Paragraph("This metric measures whether a synthetic column respects the minimum and maximum values of the real column. It returns the percentage of synthetic rows that adhere to the real boundaries.", styles['Normal']))
    content.append(Paragraph(f" ", subtitle_style2))
    content.append(Paragraph("(best) 1.0: All values in the synthetic data respect the min/max boundaries of the real data.", green_style))
    content.append(Paragraph("(worst) 0.0: No value in the synthetic data is in between the min and max value of the real data.", red_style))
    content.append(Paragraph("<br/><br/>", styles['Normal']))

    content.append(Paragraph(f"(Quality) Coverage", subtitle_style2))
    content.append(Paragraph("This metric measures whether a synthetic column covers the full range of values that are present in a real column.", styles['Normal']))
    content.append(Paragraph(f" ", subtitle_style2))
    content.append(Paragraph("(best) 1.0: The synthetic column covers the range of values present in the real column.", green_style))
    content.append(Paragraph("(worst) 0.0:  The synthetic column does not overlap at all with the range of values in the real column.", red_style))
    content.append(Paragraph("<br/><br/>", styles['Normal']))

    content.append(Paragraph(f"(Quality) Complement", subtitle_style2))
    content.append(Paragraph("This metric computes the similarity of a real column vs. a synthetic column in terms of the column shapes -- aka the marginal distribution or 1D histogram of the column.", styles['Normal']))
    content.append(Paragraph(f" ", subtitle_style2))
    content.append(Paragraph("(best) 1.0: The synthetic distribution shapes are exactly the same to the real data.", green_style))
    content.append(Paragraph("(worst) 0.0: The synthetic distribution shapes are nothing like the real data.", red_style))
    content.append(Paragraph("<br/><br/>", styles['Normal']))

    content.append(Paragraph(f"(Utility) Statistic Similarity", subtitle_style2))
    content.append(Paragraph("This metric measures the similarity between a real column and a synthetic column by comparing a summary statistic (mean, median, standard deviation).", styles['Normal']))
    content.append(Paragraph(f" ", subtitle_style2))
    content.append(Paragraph("(best) 1.0: The summary statistics are exactly the same.", green_style))
    content.append(Paragraph("(worst) 0.0: The summary statistics are completely different.", red_style))
    content.append(Paragraph("<br/><br/>", styles['Normal']))

    content.append(Paragraph(f"(Utility) Correlation Similarity", subtitle_style2))
    content.append(Paragraph("This metric measures the correlation between a pair of numerical columns and computes the similarity between the real and synthetic data -- aka it compares the trends of 2D distributions.", styles['Normal']))
    content.append(Paragraph(f" ", subtitle_style2))
    content.append(Paragraph("(best) 1.0: The pairwise correlations of the real and synthetic data are exactly the same.", green_style))
    content.append(Paragraph("(worst) 0.0: The pairwise correlations are as different as they can possibly be.", red_style))
    content.append(Paragraph("<br/><br/>", styles['Normal']))

    content.append(Paragraph(f"(Utility) ML Efficacy", subtitle_style2))
    content.append(Paragraph("This metric calculates the success of using synthetic data to perform an ML prediction task.", styles['Normal']))
    content.append(Paragraph(f" ", subtitle_style2))
    content.append(Paragraph("(best) 1.0: Given the synthetic training data, you will be able to perform ML tasks with 100% accuracy on the real data", green_style))
    content.append(Paragraph("(worst) 0.0: Given the synthetic training data, you will not be able to predict any of the real data correctly.", red_style))
    content.append(Paragraph("<br/><br/>", styles['Normal']))

    content.append(Paragraph("<br/><br/>", styles['Normal']))
    fig = reduction_vis(data, synthetic_data)
    img_data = save_figure_to_image(fig)
    img = Image(img_data, width=520, height=390)
    content.append(img)


    pdf.build(content)
    print(f"PDF report created: {pdf_file}")
