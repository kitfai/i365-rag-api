import json
import re
from typing import Any, Dict, List, Union
import json
import re
from langflow.custom import Component
from langflow.io import HandleInput, Output
from langflow.schema.message import Message

class ExtractionValidator(Component):
    """
    A custom component to validate the quality of a JSON extraction from an LLM.
    It robustly finds and parses a JSON block from the model's raw output,
    then checks for a known document type and the completeness of the extracted
    fields before routing the flow to a success or fallback path.
    """
    display_name = "Extraction Validator"
    description = "Checks the quality of an LLM's JSON extraction and routes the flow accordingly."
    icon = "TestTube"

    inputs = [
        HandleInput(
            name="llm_output",
            display_name="LLM Output",
            info="The raw text output from the language model, which may contain JSON.",
            input_types=["Message"],
            required=True,
        ),
    ]

    outputs = [
        Output(display_name="Success", name="success_output", method="route_flow"),
        Output(display_name="Fallback", name="fallback_output", method="route_flow"),
    ]
    
    def route_flow(self) -> tuple[Message, Message]:
        """
        Parses the LLM output and decides whether it's a success or requires fallback.
        Returns a tuple where the first element is for the 'Success' path and the
        second is for the 'Fallback' path. Only one will contain the message.
        """
        llm_message = self.llm_output
        if not llm_message or not llm_message.text:
            self.status = "Fallback: Empty input received."
            return (None, llm_message)

        try:
            # --- Step 1: Clean and Parse the JSON ---
            # Remove common markdown artifacts like 
            cleaned_output = re.sub(r"\s*```json\s*```", "", llm_message.text)
            json_str = self._extract_json_block(cleaned_output)
            if not json_str:
                raise ValueError("No JSON block found in the output.")
        except Exception as e:
            self.status = f"Failed to parse JSON: {e}"
            return (None, llm_message)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            self.status = f"Invalid JSON: {e}"
            return (None, llm_message)    


'''
based on uploadlangflow.json to enhance the extraction validation based on the suggestions below:

Give the full code for ExtractionValidator Component. Based on

Reliably find the JSON block, even if it's surrounded by other text or markdown. 2.Check if the model's output is valid JSON. 3.Verify that the document_type is not 'Unknown'. 4.Count how many of the expected fields are missing or incomplete. 5.Route the flow to a "Success" or "Fallback" path based on these quality checks.
** Responds in batches which each batch indicate the start and end of the response batch**

example llm output as below { "message": "<think>\nOkay, let's tackle this query step by step. The user wants me to classify the document and extract specific fields based on predefined rules.\n\nFirst, I need to determine if this is an 'Invoice', 'Payment Instruction', or 'Unknown'. Looking at the content, it starts with "THIS IS A COMPUTER GENERATED INVOICE" which clearly indicates that the document type is an invoice. There's also a mention of invoice number and payment details, so classification should be straightforward here.\n\nNow, moving on to field extraction for invoices:\n- Particulars_of_Buyer: The buyer information appears at the top with "MUHAMEED AZAM BIN ALISAN" and "SABRINA BINTI SULEMAN". This seems to be the person(s) involved in the payment.\n- invoice_number: I can see "Invoice No : PB10003102", so that's clear.\n- Date: The document mentions "Date 01/06/2022" which likely refers to when the invoice was issued. Since it says "Date of the invoice is issued," this should be extracted as the date field.\n- Project: It explicitly states "Project : MELAWATI HILLS".\n- Lot_No: Listed as "Lot No : A-08-03".\n- Selling_Price: The text includes "Selling Price : RM1,466,800.00", so that's straightforward.\n- Amount_Due_From_Buyer and Amount_Due_For_Payment: These are a bit tricky because the document mentions "Amount Due From Buyer" but in the payment section, it says "Amount Due for Payment". However, looking at the context, there is an explicit line "Amount Due For Payment : 146,680.00", which matches the extraction rule.\n\nI need to be careful here because sometimes these terms can overlap or have similar wording. But in this case, I think it's clear that both fields are present and should be extracted as specified.\n</think>\njson\n{\n \"document_type\": \"Invoice\",\n \"Particulars_of_Buyer\": \"MUHAMEED AZAM BIN ALISAN SABRINA BINTI SULEMAN\",\n \"invoice_number\": \"PB10003102\",\n \"Date\": \"01/06/2022\",\n \"Project\": \"MELAWATI HILLS\",\n \"Lot_No\": \"A-08-03\",\n \"Selling_Price\": \"RM1,466,800.00\",\n \"Amount_Due_From_Buyer\": \"RM 146,680.00\",\n \"Amount_Due_For_Payment\": \"RM 146,680.00\"\n}\n" }

expected fields as below: "document_type": "Invoice", "Particulars_of_Buyer": "MUHAMEED AZAM BIN ALISAN SABRINA BINTI SULEMAN", "invoice_number": "PB10003102", "Date": "01/06/2022", "Project": "MELAWATI HILLS", "Lot_No": "A-08-03", "Selling_Price": "RM1,466,800.00", "Amount_Due_From_Buyer": "RM 146,680.00", "Amount_Due_For_Payment": "RM 146,680.00" }

'''

class ExtractionValidator:
    """
    A component to validate the output of a language model for document extraction.

    This validator performs several checks:
    1. Reliably finds and parses a JSON block from the model's output string.
    2. Checks if the parsed JSON is valid.
    3. Verifies that the 'document_type' field is not 'Unknown'.
    4. Counts how many of the expected fields are missing or have incomplete values.
    5. Provides a routing decision ('Success' or 'Fallback') based on these checks.
    """

    def __init__(self, missing_fields_threshold: int = 2):
        """
        Initializes the validator.

        Args:
            missing_fields_threshold (int): The maximum number of missing or incomplete
                                            fields allowed for a 'Success' outcome.
        """
        self.missing_fields_threshold = missing_fields_threshold

    def _extract_json_block(self, text: str) -> Union[str, None]:
        """
        Reliably extracts a JSON block from a string, even if surrounded by text or markdown.

        Args:
            text: The raw string output from the language model.

        Returns:
            The extracted JSON string, or None if no valid block is found.
        """
        # Pattern to find JSON within ```json ... ```
        match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: find the largest JSON object in the text by locating the first '{' and last '}'
        try:
            start = text.index('{')
            end = text.rindex('}') + 1
            return text[start:end]
        except ValueError:
            return None

    def validate(self, llm_output: str, expected_fields: List[str]) -> Dict[str, Any]:
        """
        Validates the LLM output and provides a routing decision.

        Args:
            llm_output: The raw string from the language model.
            expected_fields: A list of field names expected in the JSON output.

        Returns:
            A dictionary containing the validation status, parsed data,
            and any error information.
        """
        # 1. Reliably find the JSON block
        json_str = self._extract_json_block(llm_output)
        if not json_str:
            return {
                "status": "Fallback",
                "data": None,
                "error_message": "No JSON block found in the output.",
                "missing_fields_count": len(expected_fields),
            }

        # 2. Check if the model's output is valid JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return {
                "status": "Fallback",
                "data": None,
                "error_message": f"Invalid JSON: {e}",
                "missing_fields_count": len(expected_fields),
            }

        # 3. Verify that the document_type is not 'Unknown'
        doc_type = data.get("document_type")
        if not doc_type or doc_type.lower() == "unknown":
            return {
                "status": "Fallback",
                "data": data,
                "error_message": f"Document type is '{doc_type or 'missing'}'",
                "missing_fields_count": len(expected_fields) - (1 if "document_type" in data else 0),
            }

        # 4. Count how many of the expected fields are missing or incomplete
        missing_fields_count = 0
        incomplete_values = [None, "", "N/A", "Not available", "Not specified"]
        for field in expected_fields:
            if field not in data or data.get(field) in incomplete_values:
                missing_fields_count += 1

        # 5. Route the flow to a "Success" or "Fallback" path
        if missing_fields_count > self.missing_fields_threshold:
            return {
                "status": "Fallback",
                "data": data,
                "error_message": f"{missing_fields_count} fields are missing or incomplete, which exceeds the threshold of {self.missing_fields_threshold}.",
                "missing_fields_count": missing_fields_count,
            }

        return {
            "status": "Success",
            "data": data,
            "error_message": None,
            "missing_fields_count": missing_fields_count,
        }

# Example Usage:
if __name__ == '__main__':
    # The user-provided example LLM output
    llm_output_example = """
{
  "message": "<think>\\nOkay, let's tackle this query step by step. The user wants me to classify the document and extract specific fields based on predefined rules.\\n\\nFirst, I need to determine if this is an 'Invoice', 'Payment Instruction', or 'Unknown'. Looking at the content, it starts with \\"THIS IS A COMPUTER GENERATED INVOICE\\" which clearly indicates that the document type is an invoice. There's also a mention of invoice number and payment details, so classification should be straightforward here.\\n\\nNow, moving on to field extraction for invoices:\\n- Particulars_of_Buyer: The buyer information appears at the top with \\"MUHAMEED AZAM BIN ALISAN\\" and \\"SABRINA BINTI SULEMAN\\". This seems to be the person(s) involved in the payment.\\n- invoice_number: I can see \\"Invoice No : PB10003102\\", so that's clear.\\n- Date: The document mentions \\"Date 01/06/2022\\" which likely refers to when the invoice was issued. Since it says \\"Date of the invoice is issued,\\" this should be extracted as the date field.\\n- Project: It explicitly states \\"Project : MELAWATI HILLS\\".\\n- Lot_No: Listed as \\"Lot No : A-08-03\\".\\n- Selling_Price: The text includes \\"Selling Price : RM1,466,800.00\\", so that's straightforward.\\n- Amount_Due_From_Buyer and Amount_Due_For_Payment: These are a bit tricky because the document mentions \\"Amount Due From Buyer\\" but in the payment section, it says \\"Amount Due for Payment\\". However, looking at the context, there is an explicit line \\"Amount Due For Payment : 146,680.00\\", which matches the extraction rule.\\n\\nI need to be careful here because sometimes these terms can overlap or have similar wording. But in this case, I think it's clear that both fields are present and should be extracted as specified.\\n</think>\\n```json\\n{\\n    \\"document_type\\": \\"Invoice\\",\\n    \\"Particulars_of_Buyer\\": \\"MUHAMEED AZAM BIN ALISAN SABRINA BINTI SULEMAN\\",\\n    \\"invoice_number\\": \\"PB10003102\\",\\n    \\"Date\\": \\"01/06/2022\\",\\n    \\"Project\\": \\"MELAWATI HILLS\\",\\n    \\"Lot_No\\": \\"A-08-03\\",\\n    \\"Selling_Price\\": \\"RM1,466,800.00\\",\\n    \\"Amount_Due_From_Buyer\\": \\"RM 146,680.00\\",\\n    \\"Amount_Due_For_Payment\\": \\"RM 146,680.00\\"\\n}\\n```"
}
    """
    # The message key contains the actual output string
    llm_output_str = json.loads(llm_output_example)["message"]

    # The user-provided expected fields
    expected_fields_list = [
        "document_type",
        "Particulars_of_Buyer",
        "invoice_number",
        "Date",
        "Project",
        "Lot_No",
        "Selling_Price",
        "Amount_Due_From_Buyer",
        "Amount_Due_For_Payment"
    ]

    # --- Test Case 1: Successful Extraction ---
    print("--- Test Case 1: Successful Extraction ---")
    validator = ExtractionValidator(missing_fields_threshold=2)
    result = validator.validate(llm_output_str, expected_fields_list)
    print(json.dumps(result, indent=2))
    print("-" * 40)

    # --- Test Case 2: Fallback due to Unknown document_type ---
    print("--- Test Case 2: Fallback due to Unknown document_type ---")
    llm_output_unknown = llm_output_str.replace('"Invoice"', '"Unknown"')
    result_unknown = validator.validate(llm_output_unknown, expected_fields_list)
    print(json.dumps(result_unknown, indent=2))
    print("-" * 40)

    # --- Test Case 3: Fallback due to missing fields ---
    print("--- Test Case 3: Fallback due to missing fields ---")
    llm_output_missing = llm_output_str.replace(',\\n    \\"Lot_No\\": \\"A-08-03\\"', '')
    llm_output_missing = llm_output_missing.replace(',\\n    \\"Project\\": \\"MELAWATI HILLS\\"', '')
    llm_output_missing = llm_output_missing.replace(',\\n    \\"Date\\": \\"01/06/2022\\"', '')
    result_missing = validator.validate(llm_output_missing, expected_fields_list)
    print(json.dumps(result_missing, indent=2))
    print("-" * 40)

    # --- Test Case 4: Fallback due to invalid JSON ---
    print("--- Test Case 4: Fallback due to invalid JSON ---")
    llm_output_invalid_json = llm_output_str.replace('}', '', 2) # remove closing brace from JSON
    result_invalid = validator.validate(llm_output_invalid_json, expected_fields_list)
    print(json.dumps(result_invalid, indent=2))
    print("-" * 40)

    # --- Test Case 5: No JSON block ---
    print("--- Test Case 5: No JSON block ---")
    llm_output_no_json = "<think>I could not find any information</think>"
    result_no_json = validator.validate(llm_output_no_json, expected_fields_list)
    print(json.dumps(result_no_json, indent=2))
    print("-" * 40)
