# mom_formatter.py
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from docx import Document
from docx.shared import Inches
import pandas as pd
from fpdf import FPDF


def format_mom_html(mom_text, output_path="data/mom_final.html"):
    """Generate professional HTML email-ready MoM."""
    html = f"""
<!DOCTYPE html>
<html>
<head><title>Meeting Minutes</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #34495e; margin-top: 30px; }}
ul {{ margin-left: 20px; }}
.action {{ color: #27ae60; }}
.decision {{ color: #f39c12; }}
.question {{ color: #e74c3c; }}
</style>
</head>
<body>
<h1>📋 Meeting Minutes</h1>
<pre style="white-space: pre-wrap; font-family: inherit;">{mom_text}</pre>
<p><small>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</small></p>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path

def format_mom_pdf(mom_text, output_path="data/mom_final.pdf"):

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    effective_width = pdf.w - 2 * pdf.l_margin # usable width
    for line in mom_text.split("\n"):

        if not line.strip():
            pdf.ln(5)
            continue
        pdf.multi_cell(effective_width, 8, txt=line) # use explicit width
    pdf.output(output_path)
    return output_path