import streamlit as st
import pandas as pd
from database import DDRSQL  # Assuming DDRSQL is your class for database operations
from navigation import make_sidebar

make_sidebar()

# Database connection setup
db_uri = 'postgresql://postgres:102414@localhost:5432/project'
ddrsql = DDRSQL(db_uri)

# --- Helper function to update the database ---
def update_database(ddrsql, table_name, edited_data, original_data, primary_key):
    """Update the database with changes made in the data editor."""
    conn = ddrsql.connect()
    if conn is None:
        st.error("Failed to connect to the database.")
        return

    try:
        cur = conn.cursor()

        # Handle updates for existing rows
        for idx, row in edited_data.iterrows():
            if idx < len(original_data):
                original_row = original_data.loc[idx]
                if not row.equals(original_row):
                    update_query = f"UPDATE {table_name} SET "
                    updates = []
                    for col in edited_data.columns:
                        if row[col] != original_row[col]:
                            updates.append(f"{col} = %s")
                    
                    if updates:
                        update_query += ", ".join(updates) + f" WHERE {primary_key} = %s"
                        update_values = [row[col] for col in edited_data.columns if row[col] != original_row[col]]
                        update_values.append(row[primary_key])

                        cur.execute(update_query, tuple(update_values))
                        conn.commit()

        # Handle newly added rows
        new_rows = edited_data.iloc[len(original_data):]
        for idx, new_row in new_rows.iterrows():
            insert_query = f"INSERT INTO {table_name} ({', '.join(new_row.index)}) VALUES ({', '.join(['%s'] * len(new_row))})"
            cur.execute(insert_query, tuple(new_row))
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        st.error(f"Error updating the database: {e}")
    finally:
        conn.close()

# --- Function to delete rows ---
def delete_rows(ddrsql, table_name, deleted_ids, primary_key):
    """Delete rows from the database based on their primary keys."""
    conn = ddrsql.connect()
    if conn is None:
        st.error("Failed to connect to the database.")
        return

    try:
        cur = conn.cursor()
        for row_id in deleted_ids:
            cur.execute(f"DELETE FROM {table_name} WHERE {primary_key} = %s", (row_id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        st.error(f"Error deleting rows from {table_name}: {e}")
    finally:
        conn.close()

# --- Streamlit App ---
st.title("Database Table Viewer & Editor")

# Dropdown menu for table selection
table_options = ['patients', 'diagnostic', 'images', 'user']
selected_table = st.selectbox("Select a Table to View and Edit", table_options)

primary_keys = {
    'patients': 'patient_id',
    'diagnostic': 'diag_id',
    'images': 'image_id',
    'user': 'user_id'
}

if selected_table:
    # Fetch the data
    data = ddrsql.get_data(selected_table)

    if data is not None and not data.empty:
        st.subheader(f"Data from {selected_table} Table")

        # Display editable data grid
        edited_data = st.data_editor(data, num_rows="dynamic")

        # Detect deleted rows
        original_ids = set(data[primary_keys[selected_table]])
        edited_ids = set(edited_data[primary_keys[selected_table]])
        deleted_ids = original_ids - edited_ids

        # Handle updates and deletions
        if st.button("Save Changes"):
            if not edited_data.equals(data) or deleted_ids:
                st.write("Changes detected. Updating the database...")
                
                # Update existing rows and add new rows
                update_database(ddrsql, selected_table, edited_data, data, primary_keys[selected_table])

                # Delete rows that were removed
                if deleted_ids:
                    delete_rows(ddrsql, selected_table, deleted_ids, primary_keys[selected_table])

                st.success("Database updated successfully!")
            else:
                st.write("No changes detected.")
    else:
        st.write(f"No records found in the {selected_table} table.")
else:
    st.error("Please select a table to view.")
