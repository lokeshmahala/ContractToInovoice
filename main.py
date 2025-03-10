import streamlit as st
import os
import base64
from mistralai import Mistral
import boto3
from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import time

st.set_page_config(layout="wide", page_title="Mistral OCR App", page_icon="🖥️")
st.title("Mistral OCR App")
# with st.expander("Expand Me"):
#     st.markdown("""
#     This application allows you to extract information from pdf/image based on Mistral OCR. Built by AI Anytime.
#     """)

# 1. API Key Input
api_key = st.text_input("Enter your Mistral API Key", type="password")
if not api_key:
    st.info("Please enter your API key to continue.")
    st.stop()

# Initialize session state variables for persistence
if "ocr_result" not in st.session_state:
    st.session_state["ocr_result"] = None
if "preview_src" not in st.session_state:
    st.session_state["preview_src"] = None
if "image_bytes" not in st.session_state:
    st.session_state["image_bytes"] = None

# 2. Choose file type: PDF or Image
file_type = st.radio("Select file type", ("PDF", "Image"))

# 3. Select source type: URL or Local Upload
source_type = st.radio("Select source type", ("URL", "Local Upload"))

input_url = ""
uploaded_file = None

if source_type == "URL":
    if file_type == "PDF":
        input_url = st.text_input("Enter PDF URL")
    else:
        input_url = st.text_input("Enter Image URL")
else:
    if file_type == "PDF":
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    else:
        uploaded_file = st.file_uploader("Upload an Image file", type=["jpg", "jpeg", "png"])



def upload_to_s3(file_bytes, filename):
    s3 = boto3.client("s3", aws_access_key_id="AKIASFIXC2RE2ZAFPBGV", aws_secret_access_key="MD7PmA1xZoZa5GqDXkfg4IR1AvyaFgZa4qN5sK0q")
    bucket_name = "ocr-mistral-poc2"  # Change this to your bucket name
    s3_key = f"uploads/{filename}"

    s3.put_object(Bucket=bucket_name, Key=s3_key, Body=file_bytes, ContentType="application/pdf",ACL="public-read")
    
    return f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"


# 4. Process Button & OCR Handling
if st.button("Process"):
    if source_type == "URL" and not input_url:
        st.error("Please enter a valid URL.")
    elif source_type == "Local Upload" and not uploaded_file:
        st.error("Please upload a file.")
    else:
        client = Mistral(api_key=api_key)
        # Prepare the document payload and preview source based on type & source.
        if file_type == "PDF":
            if source_type == "URL":
                document = {
                    "type": "document_url",
                    "document_url": input_url
                }
                preview_src = input_url
            else:
                # print("Uploaded file: ", uploaded_file)
                # print(uploaded_file)
                file_bytes = uploaded_file.read()
                # print("File bytes: ", file_bytes)
                # encoded_pdf = base64.b64encode(file_bytes).decode("utf-8")
                file_url = upload_to_s3(file_bytes, uploaded_file.name)
                # print("File URL: ", file_url)
                document = {
                    "type": "document_url",
                    "document_url": file_url
                }
                # print("Document: ", document)
                preview_src = file_url 
                # print("Preview src: ", preview_src)
                # f"data:application/pdf;base64,{encoded_pdf}"
                # temp_pdf_path = f"/tmp/{uploaded_file.name}"  # Change this path as needed
                # with open(temp_pdf_path, "wb") as f:
                #     f.write(uploaded_file.read())

                # document = {
                #     "type": "document_url",  # Ensure we pass the correct type
                #     "document_url": f"file://{temp_pdf_path}"  # If Mistral supports local paths
                # }
                # preview_src = temp_pdf_path  # Adjust how the preview is displayed
        else:  # file_type == "Image"
            if source_type == "URL":
                document = {
                    "type": "image_url",
                    "image_url": input_url
                }
                preview_src = input_url
            else:
                file_bytes = uploaded_file.read()
                mime_type = uploaded_file.type
                encoded_image = base64.b64encode(file_bytes).decode("utf-8")
                document = {
                    "type": "image_url",
                    "image_url": f"data:{mime_type};base64,{encoded_image}"
                }
                preview_src = f"data:{mime_type};base64,{encoded_image}"
                st.session_state["image_bytes"] = file_bytes 

        with st.spinner("Processing the document..."):
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document=document,
                include_image_base64=True
            )
            # Extract OCR results by joining markdown from each OCRPageObject
            try:
                if hasattr(ocr_response, "pages"):
                    pages = ocr_response.pages
                elif isinstance(ocr_response, list):
                    pages = ocr_response
                else:
                    pages = []
                result_text = "\n\n".join(page.markdown for page in pages)
                if not result_text:
                    result_text = "No result found."
            except Exception as e:
                result_text = f"Error extracting result: {e}"
            st.session_state["ocr_result"] = result_text
            st.session_state["preview_src"] = preview_src

            ocr_text = "\n\n".join(page.markdown for page in ocr_response.pages) if hasattr(ocr_response, "pages") else "No text extracted."

            # Display extracted text
            st.text_area("Extracted OCR Text", ocr_text, height=300)

            # Store OCR text in session state
            st.session_state["ocr_text"] = ocr_text

# 3. LlamaIndex Integration (Only if OCR text exists)
if "ocr_text" in st.session_state and st.button("Analyze with LlamaIndex"):
    ocr_text = st.session_state["ocr_text"]
    
    print("OCR Text: ", ocr_text)
    # Initialize LlamaIndex Settings
    # Settings.llm = MistralAI(model="mistral-medium", api_key=api_key)
    # Settings.embed_model = MistralAIEmbedding(model_name="mistral-embed", api_key=api_key)
    Settings.llm = Ollama(model="llama3.2:latest", request_timeout=120.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("Settings: ", Settings)
    # Convert OCR text into LlamaIndex Document
    document = Document(text=ocr_text)
    print("Document: ", document)
    
    # Create Vector Store Index
    index = VectorStoreIndex.from_documents([document])

    print("Index: ", index)
    # Create Query Engine
    query_engine = index.as_query_engine(similarity_top_k=2)

    print("Query Engine: ", query_engine)
    # User Query Input
    # user_query = st.text_input("Ask a question based on the extracted text:")
    # user_query = "Extract the full legal name of the customer (the entity entering into the agreement) as stated in the contract. Ensure that you retrieve the exact name listed under the section explicitly identifying the Customer. If multiple customer names are mentioned, select the one that is formally defined as the contracting party."
    # if user_query:
    #     print("User Query: ", user_query)
    #     with st.spinner("Searching for answers..."):
    #         response = query_engine.query(user_query)
    #         print("Response: ", response)
    #     st.subheader("Response:")
    #     st.write(response.response)

    # Define multiple queries
    user_queries = [
        "Extract the full legal name of the customer (the entity entering into the agreement) as stated in the contract. Ensure that you retrieve the exact name listed under the section explicitly identifying the Customer. If multiple customer names are mentioned, select the one that is formally defined as the contracting party."
        "Identify the individual designated as the principal contact for the customer regarding billing matters. Extract their full name, job title (if provided), and email address. Look for sections labeled Customer Principal Contact or any billing-related contact details.",
        "Extract the full shipping address (including company name, street address, city, state, postal code, and country) specifically identified as the “Ship To” or “Tax Address” in the contract. Ensure you retrieve the correct address designated for tax purposes.",
        "Identify and extract the date when the master agreement was executed between the parties. Look for phrases such as Agreement Date, Master Agreement Date, or Effective Date of Agreement. Ensure that the date represents the execution of the primary contract rather than any amendments or order renewals.",
        "Extract the official start date of the order term. Look for sections labeled Order Term Start Date or any statements specifying when the contractual obligations begin. If multiple start dates exist, prioritize the general order term start date unless otherwise indicated.",
        "Extract the official end date of the order term. Look for sections labeled Order Term End Date or any statements defining the expiration of contractual obligations. Ensure that this date corresponds to the conclusion of the overall agreement.",
        "Summarize the renewal terms of the agreement. Extract details on whether the contract automatically renews, the duration of the renewal period, any fee adjustments, and conditions for renewal or termination. Look for keywords such as Renewal Term, Renewal Conditions, or Automatic Renewal.",
        "Extract the email address designated for invoice-related communications. Look for sections labeled Invoice Email Address or any references to electronic billing contacts.",
        "Identify the individual or department responsible for accounts payable. Extract their full name, job title (if available), and contact details. Look for terms such as Accounts Payable Contact or Billing Contact.",
        "Extract the tax identification number (TIN) associated with the customer. This may be listed under “Tax ID,” “VAT Number,” or “Taxpayer Identification.” If no tax ID is explicitly mentioned, state that it is not provided in the contract.",
        "Determine whether a purchase order (PO) is required for billing. Extract the relevant section specifying if the contract requires a PO to process invoices. Return the response as either Yes or No based on the contract language",
        "Extract the purchase order number (PO Number) if provided in the contract. Look for a section labeled Purchase Order Number or any references to PO-related billing requirements. If no PO number is provided, return N/A.",
        "Extract and summarize the invoice terms, including how invoices are issued, who invoices whom, and any relevant conditions such as third-party invoicing. Look for sections titled “Invoice Terms” or descriptions of invoicing procedures",
        "Extract and summarize the payment terms, including payment due dates, responsible payment entities, and any stipulations on refunds or terminations. Look for sections titled Payment Terms or contractual obligations related to financial transactions.",
        "Extract the currency in which all financial transactions under this contract are conducted. Look for mentions of Currency or monetary references specifying USD, EUR, etc.",
        "Extract any additional terms mentioned in the contract, particularly clauses related to service availability, special provisions, or unique contractual conditions. Look for a section labeled Additional Terms or clauses specifying additional obligations beyond the core agreement.",
        "Extract a structured list of all products and services included in the contract. For each product, retrieve: Product name, SKU (if available), Quantity (annual or total), Price (if provided), Start date, End date",
        "Extract the full billing schedule and corresponding payment amounts for each billing period. Look for details on how and when payments are structured throughout the contract term. Identify specific yearly totals and any additional billing conditions.",
        "Extract the total value of the contract for the entire term, including all charges and fees. Look for phrases like Total Term Amount, Grand Total, or Contract Value. Ensure that all numbers are correctly captured, including additional costs such as marketplace fees or penalties if applicable."
    ]

    results = []
    with st.spinner("Searching for answers..."):
        for query in user_queries:
            response = query_engine.query(query)
            results.append((query, response.response))
            time.sleep(1)  # Add a delay to avoid rate limits
    # Display results
    st.subheader("Extracted Information:")
    for query, answer in results:
        st.markdown(f"**Q: {query}**")
        st.write(f"**A:** {answer}")
        st.markdown("---")  # Add a separator for clarity

# 5. Display Preview and OCR Result if available
# if st.session_state["ocr_result"]:
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("Preview")
#         if file_type == "PDF":
#             # Embed PDF via iframe
#             pdf_embed_html = (
#                 f'<iframe src="{st.session_state["preview_src"]}" width="100%" '
#                 f'height="800" frameborder="0"></iframe>'
#             )
#             st.markdown(pdf_embed_html, unsafe_allow_html=True)
#         else:
#             # For images, display using st.image
#             if source_type == "Local Upload" and st.session_state["image_bytes"]:
#                 st.image(st.session_state["image_bytes"])
#             else:
#                 st.image(st.session_state["preview_src"])
    
#     with col2:
#         st.subheader("OCR Result")
#         st.write(st.session_state["ocr_result"])
#         # Create a custom download link for OCR result
#         b64 = base64.b64encode(st.session_state["ocr_result"].encode()).decode()
#         href = f'<a href="data:file/txt;base64,{b64}" download="ocr_result.txt">Download OCR Result</a>'
#         st.markdown(href, unsafe_allow_html=True)
