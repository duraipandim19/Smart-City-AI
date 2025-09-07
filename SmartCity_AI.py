import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import PyPDF2  # For PDF handling
import re

# Load model and tokenizer
# NOTE: This model is very large. It may require a significant amount of GPU memory.
model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, max_length=1024, max_new_tokens=None):
    """
    Generates a text response from the loaded language model based on a prompt.
    Now includes a flexible max_new_tokens parameter to control output length.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generation_params = {
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    if max_new_tokens is not None:
        generation_params["max_new_tokens"] = max_new_tokens
    else:
        generation_params["max_length"] = max_length

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_params
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

# --- New Functions for Smart City Assistant ---

def summarize_policy(file):
    """
    Handles the Policy Search & Summarization use case.
    Reads a PDF, summarizes it, and provides a concise version.
    """
    if file is None:
        return "Please upload a policy document (PDF)."

    try:
        pdf_reader = PyPDF2.PdfReader(file.name)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        prompt = f"Summarize the following city policy document into a concise, citizen-friendly version, highlighting key points and implications.\n\nDocument:\n{text}"
        summary = generate_response(prompt, max_length=2000)
        return summary
    except Exception as e:
        return f"An error occurred while processing the PDF: {e}"

def submit_feedback(name, location, issue_type, description):
    """
    Handles the Citizen Feedback Reporting use case.
    Logs the report and uses the LLM to categorize it.
    """
    if not description:
        return "Please provide a description of the issue."

    try:
        # Use LLM to categorize the issue. We'll use max_new_tokens for this.
        categorization_prompt = f"Categorize the following citizen report into a single, concise category (e.g., 'Water', 'Traffic', 'Waste', 'Infrastructure').\n\nReport: {description}"
        category = generate_response(categorization_prompt, max_new_tokens=20).strip()
        
        # Simulate logging the report
        report_status = f"Thank you, {name}! Your report has been logged. We've categorized it as: '{category}'. A team will review it shortly."
        return report_status
    except Exception as e:
        return f"An error occurred while submitting the report: {e}"

def forecast_kpis(file):
    """
    Handles the KPI Forecasting use case.
    Accepts a CSV, performs a simple forecast, and displays the results.
    """
    if file is None:
        return "Please upload a CSV file with KPI data."

    try:
        df = pd.read_csv(file.name)
        # Check for required columns
        if 'date' not in df.columns or 'kpi' not in df.columns:
            return "CSV must contain 'date' and 'kpi' columns."
            
        df['date'] = pd.to_datetime(df['date'])
        
        last_kpi = df['kpi'].iloc[-1]
        
        # Simple projection for the next 3 periods based on a linear trend
        forecast_period = 3
        forecast_growth = (df['kpi'].iloc[-1] / df['kpi'].iloc[-2] - 1) if len(df) > 1 else 0.05
        
        forecast = [last_kpi * (1 + (i+1) * forecast_growth) for i in range(forecast_period)]
        
        forecast_df = pd.DataFrame({
            "Period": range(1, forecast_period + 1),
            "Forecasted KPI": forecast
        })
        
        return forecast_df.to_string()
    except Exception as e:
        return f"An error occurred during forecasting: {e}. Please ensure the CSV is formatted correctly."


# Create Gradio interface
with gr.Blocks(css=".gradio-container { max-width: 1000px; margin: auto; padding: 20px; }") as app:
    gr.Markdown("# Sustainable Smart City Assistant")
    gr.Markdown("An AI-powered platform for urban sustainability, governance, and citizen engagement.")

    with gr.Tabs():
        with gr.TabItem("Policy & Document Analysis"):
            gr.Markdown("### Policy Search & Summarization")
            file_upload = gr.File(label="Upload Policy Document (PDF)")
            summarize_btn = gr.Button("Summarize Document")
            summary_output = gr.Textbox(label="Concise Summary", interactive=False, lines=10)
            
            summarize_btn.click(
                summarize_policy,
                inputs=file_upload,
                outputs=summary_output
            )

        with gr.TabItem("Citizen Hub"):
            gr.Markdown("### Citizen Feedback Reporting")
            with gr.Row():
                name_input = gr.Textbox(label="Your Name", placeholder="Your Name")
                location_input = gr.Textbox(label="Location", placeholder="Street, landmark, etc.")
            
            issue_type_dropdown = gr.Dropdown(label="Issue Type", choices=["Water", "Traffic", "Waste", "Infrastructure", "Other"], value="Other")
            description_text = gr.Textbox(label="Describe the Issue", lines=5, placeholder="e.g., A burst water pipe on Oak Street...")
            
            submit_feedback_btn = gr.Button("Submit Report")
            feedback_output = gr.Textbox(label="Report Status", interactive=False)

            submit_feedback_btn.click(
                submit_feedback,
                inputs=[name_input, location_input, issue_type_dropdown, description_text],
                outputs=feedback_output
            )

        with gr.TabItem("City Dashboard"):
            gr.Markdown("### KPI Forecasting")
            kpi_file_upload = gr.File(label="Upload KPI Data (CSV)")
            forecast_btn = gr.Button("Forecast Next Year's KPI")
            forecast_output = gr.Textbox(label="Forecast Results", interactive=False, lines=10)
            
            forecast_btn.click(
                forecast_kpis,
                inputs=kpi_file_upload,
                outputs=forecast_output
            )
            
app.launch(share=True)