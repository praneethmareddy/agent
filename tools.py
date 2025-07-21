import os
import json
import re
import pandas as pd
import numpy as np
import uuid
from config import PROCESSED_DIR, UPLOAD_DIR, FOLDERS, COLUMN_MAP, EMBED_MODEL
from typing import List, Dict, Any, Optional, Union, Sequence
from langchain_core.tools import tool
from faiss_utils import _faiss_retrieve
import shutil
# ===== RETRIEVAL TOOLS (More Flexible) =====
standard_template= "x.xlsx"
@tool
def search_documents(query: str, doc_type: str = "auto") -> str:
    """
    Universal document search tool. Auto-detects document type or use specific type.
    
    Args:
        query: Search query
        doc_type: "log", "template", "ciq", "standard_ciq", "master_template", or "auto"
    
    Use for ANY search queries like:
    - "search for errors in logs"
    - "find template with specific config"
    - "look up cell information"
    - "search all documents for X"
    """
    if doc_type == "auto":
        # Auto-detect based on query content
        query_lower = query.lower()
        if any(word in query_lower for word in ["log", "error", "exception", "debug"]):
            doc_type = "log"
        elif any(word in query_lower for word in ["template", "config", "ne"]):
            doc_type = "template"
        elif any(word in query_lower for word in ["ciq", "cell", "enodeb", "pci"]):
            doc_type = "ciq"
        elif any(word in query_lower for word in ["standard", "schema"]):
            doc_type = "standard_ciq"
        elif any(word in query_lower for word in ["master"]):
            doc_type = "master_template"
        else:
            doc_type = "ciq"  # Default fallback
    
    return _faiss_retrieve(query, doc_type)

# ===== FILE MANAGEMENT TOOLS =====

@tool
def analyze_file(filename: str, operation: str = "basic") -> str:
    """
    Comprehensive file analysis tool.
    
    Args:
        filename: Name of uploaded file
        operation: "basic", "detailed", "structure", "preview", "stats"
    
    Use for:
    - "analyze this file"
    - "show file structure"
    - "preview file content"
    - "get file statistics"
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"
        
        if filename.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            return f"Unsupported file format: {filename}"

        analysis = {
            'filename': filename,
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.astype(str).to_dict()
        }

        if operation in ["detailed", "stats"]:
            analysis.update({
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
            })

        if operation in ["preview", "detailed"]:
            analysis['sample_data'] = df.head(5).to_dict(orient='records')

        if operation == "structure":
            analysis['column_info'] = {
                col: {
                    'type': str(df[col].dtype),
                    'unique_count': df[col].nunique(),
                    'null_count': df[col].isnull().sum(),
                    'sample_values': df[col].dropna().head(3).tolist()
                }
                for col in df.columns
            }

        return json.dumps(analysis, indent=2, default=str)

    except Exception as e:
        return f"Error analyzing file: {str(e)}"

@tool
def upload_file_to_category(filename: str, category: str, description: str = "") -> str:
    """
    Upload and categorize files for processing.
    
    Args:
        filename: Name of uploaded file
        category: "log", "template", "ciq", "standard_ciq", "master_template"
        description: Optional description
    
    Use for:
    - "upload this as a log file"
    - "categorize this file as template"
    - "save this CIQ file"
    """
    try:
        source_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(source_path):
            return f"Error: File not found at {source_path}"
        
        if category not in FOLDERS:
            return f"Error: Invalid category. Available: {list(FOLDERS.keys())}"
        
        dest_dir = FOLDERS[category]
        dest_path = os.path.join(dest_dir, filename)
        
        # Copy file to appropriate category folder
        import shutil
        shutil.copy2(source_path, dest_path)
        
        # Log the upload
        upload_log = {
            'filename': filename,
            'category': category,
            'description': description,
            'upload_time': pd.Timestamp.now().isoformat(),
            'file_size': os.path.getsize(source_path)
        }
        
        return f"File '{filename}' successfully uploaded to {category} category. {json.dumps(upload_log, indent=2)}"
    
    except Exception as e:
        return f"Error uploading file: {str(e)}"

# ===== CIQ PROCESSING TOOLS =====

@tool
def convert_ciq_format(
    filename: str,
    conversion_type: str = "column_to_row",
    base_cols: Optional[List[str]] = None,
    save_result: bool = True
) -> Dict[str, Any]:
    """
    Convert CIQ between different formats.
    
    Args:
        filename: Name of CIQ file to convert
        conversion_type: "column_to_row", "row_to_column"
        base_cols: Base columns to preserve (auto-detect if None)
        save_result: Whether to save converted file
    
    Use for:
    - "convert this CIQ to row format"
    - "change CIQ format"
    - "transform CIQ structure"
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            return {"error": f"File not found at {file_path}"}
        
        if filename.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            return {"error": f"Unsupported file format: {filename}"}

        if conversion_type == "column_to_row":
            converted_data = _convert_column_to_row(df, base_cols)
        else:
            return {"error": f"Conversion type '{conversion_type}' not yet implemented"}

        result = {
            "original_shape": df.shape,
            "converted_shape": (len(converted_data), len(converted_data[0]) if converted_data else 0),
            "converted_data": converted_data
        }

        if save_result:
            file_id = uuid.uuid4().hex[:8]
            output_filename = f"converted_{conversion_type}_{file_id}.xlsx"
            # Also copy to UPLOAD_DIR for use in standardization
            

            

            output_path = os.path.join(PROCESSED_DIR, output_filename)

            
            pd.DataFrame(converted_data).to_excel(output_path, index=False)
            
            converted_upload_path = os.path.join(UPLOAD_DIR, output_filename)
            shutil.copy2(output_path, converted_upload_path)
            result["download_link"] = f"https://3e516a7f176b.ngrok-free.app/download/{output_filename}"
            result["saved_as"] = output_filename
            result["converted_filename"] = output_filename  # to pass into standardization
            result["upload_path"] = converted_upload_path   # optional, for debug/logging

        return result

    except Exception as e:
        return {"error": f"Error converting CIQ: {str(e)}"}



# === HELPER ===

# === EMBEDDING-BASED MAPPER ===

@tool
def modify_standard_ciq(
    operation: str,
    template_name: str = "x.xlsx",
    modifications: Optional[Dict[str, Any]] = None,
    save_as: Optional[str] = None
) -> Dict[str, Any]:
    """
    Modify standard CIQ templates.
    
    Args:
        operation: "add_column", "remove_column", "rename_column", "reorder_columns", "update_values"
        template_name: Name of template to modify
        modifications: Dictionary of modifications to apply
        save_as: New filename to save (optional)
    
    Use for:
    - "add column to standard CIQ"
    - "modify standard template"
    - "update CIQ schema"
    - "customize standard format"
    """
    try:
        template_path = os.path.join(FOLDERS["standard_ciq"], template_name)
        if not os.path.exists(template_path):
            return {"error": f"Template '{template_name}' not found"}

        df = pd.read_excel(template_path)
        original_columns = df.columns.tolist()

        # Fix: Ensure modifications is a dict
        if modifications is None:
            modifications = {}

        if operation == "add_column":
            new_cols = modifications.get("columns", [])
            for col in new_cols:
                if col not in df.columns:
                    df[col] = ""

        elif operation == "remove_column":
            cols_to_remove = modifications.get("columns", [])
            df = df.drop(columns=[col for col in cols_to_remove if col in df.columns])

        elif operation == "rename_column":
            rename_map = modifications.get("rename_map", {})
            df = df.rename(columns=rename_map)

        elif operation == "reorder_columns":
            new_order = modifications.get("column_order", [])
            if all(col in df.columns for col in new_order):
                remaining_cols = [col for col in df.columns if col not in new_order]
                df = df[new_order + remaining_cols]

        elif operation == "update_values":
            updates = modifications.get("updates", {})
            for col, value in updates.items():
                if col in df.columns:
                    df[col] = value

        # Save modified template
        if save_as:
            output_filename = save_as
        else:
            file_id = uuid.uuid4().hex[:8]
            output_filename = f"modified_{template_name.split('.')[0]}_{file_id}.xlsx"
        
        output_path = os.path.join(FOLDERS["standard_ciq"], output_filename)
        df.to_excel(output_path, index=False)

        # Fix: Ensure df.head(3) is a DataFrame and to_dict is called correctly
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            preview = df.head(3).to_dict(orient="records")
        else:
            preview = []

        return {
            "operation": operation,
            "original_columns": original_columns,
            "modified_columns": df.columns.tolist(),
            "shape": df.shape,
            "saved_as": output_filename,
            "download_link": f"https://3e516a7f176b.ngrok-free.app/download/{output_filename}",
            "preview": preview
        }

    except Exception as e:
        return {"error": f"Error modifying standard CIQ: {str(e)}"}

@tool
def generate_downloadable_file(
    data: Union[List[Dict], pd.DataFrame],
    filename: str,
    file_format: str = "xlsx",
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Generate downloadable files from data.
    
    Args:
        data: Data to save (list of dicts or DataFrame)
        filename: Output filename
        file_format: "xlsx", "csv", "json"
        metadata: Optional metadata to include
    
    Use for:
    - "save this data as Excel"
    - "generate download link"
    - "export processed data"
    """
    try:
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data

        file_id = uuid.uuid4().hex[:8]
        base_name = filename.split('.')[0]
        output_filename = f"{base_name}_{file_id}.{file_format}"
        output_path = os.path.join(PROCESSED_DIR, output_filename)

        if file_format == "xlsx":
            df.to_excel(output_path, index=False)
        elif file_format == "csv":
            df.to_csv(output_path, index=False)
        elif file_format == "json":
            df.to_json(output_path, orient="records", indent=2)
        else:
            return {"error": f"Unsupported format: {file_format}"}

        result = {
            "filename": output_filename,
            "format": file_format,
            "shape": df.shape,
            "download_link": f"https://3e516a7f176b.ngrok-free.app/download/{output_filename}",
            "file_size": os.path.getsize(output_path)
        }

        if metadata:
            result["metadata"] = metadata

        return result

    except Exception as e:
        return {"error": f"Error generating file: {str(e)}"}

# ===== HELPER FUNCTIONS =====

def _convert_column_to_row(df: pd.DataFrame, base_cols: Optional[List[str]] = None) -> Sequence[dict[str, Any]]:
    """Internal function for column to row conversion."""
    column_map = COLUMN_MAP
    if base_cols is None:
        base_cols = [col for col in df.columns if not any(re.match(pat, col, re.IGNORECASE) for pat in column_map.values())]

    column_matches = {}
    for logical_name, pattern in column_map.items():
        for col in df.columns:
            if re.fullmatch(pattern, col, flags=re.IGNORECASE):
                idx_match = re.search(r'\d+', col)
                idx = idx_match.group() if idx_match else "0"
                operator_match = re.match(r'(vf|tef)', col, re.IGNORECASE)
                op = operator_match.group(1).lower() if operator_match else ""
                column_matches.setdefault((op, idx), {})[logical_name] = col

    all_rows = []
    for _, row in df.iterrows():
        base_data = {col: row[col] for col in base_cols}
        for (op, idx), logical_cols in column_matches.items():
            new_row = base_data.copy()
            for logical_name, actual_col in logical_cols.items():
                new_row[logical_name] = row.get(actual_col, "")
            new_row["operator"] = op
            new_row["cell_index"] = idx
            all_rows.append(new_row)

    return pd.DataFrame(all_rows).to_dict(orient="records")

# === MAIN FUNCTION ===
def standardize_ciq(
    filename: str,
    custom_mapping: Optional[Dict[str, str]] = None,
    save_result: bool = True
) -> Dict[str, Any]:
    """
    Standardize CIQ files against a standard template.
    
    Args:
        filename: Name of CIQ file to standardize
        custom_mapping: Optional manual overrides for column mapping
        save_result: Whether to save the resulting file

    Returns:
        Dict containing standardized data, download link, preview, etc.
    """
    try:
        # Load uploaded CIQ file
        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            return {"error": f"File not found at {file_path}"}
        
        if filename.endswith('.xlsx'):
            input_df = pd.read_excel(file_path, engine='openpyxl')
        elif filename.endswith('.csv'):
            input_df = pd.read_csv(file_path)
        else:
            return {"error": f"Unsupported file format: {filename}"}

        # Load standard template
        template_path = os.path.join(FOLDERS["standard_ciq"], standard_template)
        if not os.path.exists(template_path):
            return {"error": f"Standard template '{standard_template}' not found"}

        standard_df = pd.read_excel(template_path, engine='openpyxl')

        # Run standardization
        result = _standardize_with_template(input_df, standard_df, custom_mapping)

        # Save output if required
        if save_result:
            file_id = uuid.uuid4().hex[:8]
            output_filename = f"standardized_{file_id}.xlsx"
            output_path = os.path.join(PROCESSED_DIR, output_filename)
            result["output_df"].to_excel(output_path, index=False)
            result["download_link"] = f"https://3e516a7f176b.ngrok-free.app/download/{output_filename}"
            result["saved_as"] = output_filename

        # Add markdown preview
        result["standardized_data"] = result["output_df"].to_dict(orient="records")
        try:
            result["markdown_preview"] = result["output_df"].head(5).to_markdown(index=False)
        except Exception:
            result["markdown_preview"] = "Preview not available due to formatting error."

        del result["output_df"]
        return result

    except Exception as e:
        return {"error": f"Error standardizing CIQ: {str(e)}"}

# === HELPER ===
def _standardize_with_template(
    input_df: pd.DataFrame,
    standard_df: pd.DataFrame,
    custom_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    exception_map = {
        'enodeb_id': 'neid',
        'enodeb_name': 'netype',
        'cell_id': 'cellid',
        'cell_name': 'cellname',
        'pci': 'pci',
        'tac': 'tac',
        'lcu': 'lcu',
    }

    input_cols = list(input_df.columns)
    standard_cols = list(standard_df.columns)

    mapping = map_with_semantic_and_exceptions(
        input_cols, standard_cols, threshold=0.75, exception_map=exception_map
    )

    # Apply custom overrides
    if custom_mapping:
        mapping.update(custom_mapping)

    standardized_data = {}
    for std_col in standard_cols:
        matched_cols = [col for col, mapped_to in mapping.items() if mapped_to == std_col]
        if matched_cols:
            standardized_data[std_col] = input_df[matched_cols[0]]
        else:
            standardized_data[std_col] = [pd.NA] * len(input_df)

    unmatched_cols = [col for col, mapped_to in mapping.items() if mapped_to is None]

    return {
        "output_df": pd.DataFrame(standardized_data),
        "unmatched_columns": unmatched_cols,
        "used_mapping": mapping
    }

# === EMBEDDING-BASED MAPPER ===
def map_with_semantic_and_exceptions(
    unstandard_cols: List[str],
    standard_cols: List[str],
    threshold: float = 0.75,
    exception_map: Optional[Dict[str, str]] = None
) -> Dict[str, Optional[str]]:
    exception_map = exception_map or {}

    # Compute embeddings
    un_embeds = EMBED_MODEL.encode(unstandard_cols, convert_to_numpy=True)
    std_embeds = EMBED_MODEL.encode(standard_cols, convert_to_numpy=True)

    def normalize(v): return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-10, None)
    un_embeds = normalize(un_embeds)
    std_embeds = normalize(std_embeds)

    sim_matrix = np.dot(un_embeds, std_embeds.T)
    mapping = {}

    for i, un_col in enumerate(unstandard_cols):
        best_idx = np.argmax(sim_matrix[i])
        best_score = sim_matrix[i][best_idx]

        if best_score >= threshold:
            mapping[un_col] = standard_cols[best_idx]
        elif un_col.lower() in exception_map:
            mapping[un_col] = exception_map[un_col.lower()]
        else:
            mapping[un_col] = None

    return mapping

# ===== TOOLS LIST =====
TOOLS = [
    search_documents,
    analyze_file,
    upload_file_to_category,
    convert_ciq_format,
    standardize_ciq,
    modify_standard_ciq,
    generate_downloadable_file
]