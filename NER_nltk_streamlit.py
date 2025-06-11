#!/usr/bin/env python3
"""
Streamlit Entity Linker Application

A web interface for the Entity Linker using Streamlit.
This application provides an easy-to-use interface for entity extraction,
linking, and visualization.

Author: Based on entity_linker.py
Version: 1.0
"""

import streamlit as st

# Optional authentication - only if config.yaml exists
try:
    import streamlit_authenticator as stauth
    import yaml
    from yaml.loader import SafeLoader
    import os
    
    # Check if config file exists
    if os.path.exists('config.yaml'):
        # Load configuration
        with open('config.yaml') as file:
            config = yaml.load(file, Loader=SafeLoader)

        # Setup authentication
        authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days']
        )

        name, auth_status, username = authenticator.login(location='main')

        if auth_status == False:
            st.error("Username/password is incorrect")
            st.stop()
        elif auth_status == None:
            st.warning("Please enter your username and password")
            st.stop()
        elif auth_status:
            authenticator.logout("Logout", "sidebar")
            st.sidebar.success(f"Welcome *{name}*")
    else:
        st.info("No authentication configured - running in open mode")
        
except ImportError:
    st.warning("streamlit-authenticator not installed - running without authentication")
except Exception as e:
    st.error(f"Authentication error: {e}")
    st.info("Running without authentication")

import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import io
import base64
from typing import List, Dict, Any
import sys
import os

# Import our entity linker
try:
    from entity_linker import EntityLinker
except ImportError:
    st.error("entity_linker.py not found! Please ensure it's in the same directory as this file.")
    st.stop()


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import io
import base64
from typing import List, Dict, Any
import sys
import os

# Import our entity linker
try:
    from entity_linker import EntityLinker
except ImportError:
    st.error("entity_linker.py not found! Please ensure it's in the same directory as this file.")
    st.stop()


class StreamlitEntityLinker:
    """
    Streamlit wrapper for the EntityLinker class.
    
    Provides a web interface with additional visualization and
    export capabilities for entity analysis.
    """
    
    def __init__(self):
        """Initialize the Streamlit Entity Linker."""
        self.entity_linker = EntityLinker()
        
        # Configure Streamlit page
        st.set_page_config(
            page_title="Entity Linker",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state
        if 'entities' not in st.session_state:
            st.session_state.entities = []
        if 'processed_text' not in st.session_state:
            st.session_state.processed_text = ""
        if 'html_content' not in st.session_state:
            st.session_state.html_content = ""
        if 'analysis_title' not in st.session_state:
            st.session_state.analysis_title = "text_analysis"
        if 'last_processed_hash' not in st.session_state:
            st.session_state.last_processed_hash = ""

    @st.cache_data
    def cached_extract_entities(_self, text: str) -> List[Dict[str, Any]]:
        """Cached entity extraction to avoid reprocessing same text."""
        return _self.entity_linker.extract_entities(text)
    
    @st.cache_data  
    def cached_link_to_wikidata(_self, entities_json: str) -> str:
        """Cached Wikidata linking."""
        import json
        entities = json.loads(entities_json)
        linked_entities = _self.entity_linker.link_to_wikidata(entities)
        return json.dumps(linked_entities)
    
    @st.cache_data
    def cached_link_to_britannica(_self, entities_json: str) -> str:
        """Cached Britannica linking."""
        import json
        entities = json.loads(entities_json)
        linked_entities = _self.entity_linker.link_to_britannica(entities)
        return json.dumps(linked_entities)
        """Render the application header."""
        st.title("Entity Linker")
        st.markdown("""
        **Extract and link named entities from text to external knowledge bases**
        
        This tool uses NLTK for Named Entity Recognition (NER) and links entities to:
        - **Wikidata**: Structured knowledge base
        - **Britannica**: Encyclopedia articles  
        - **OpenStreetMap**: Geographic coordinates
        """)

    def render_sidebar(self):
        """Render the sidebar with configuration options."""
        st.sidebar.header("Configuration")
        
        # Entity type filters
        st.sidebar.subheader("Entity Type Filters")
        entity_types = ["PERSON", "ORGANIZATION", "GPE", "LOCATION", "FACILITY", "ADDRESS"]
        selected_types = st.sidebar.multiselect(
            "Show Entity Types",
            entity_types,
            default=entity_types,
            help="Select which entity types to display in results"
        )
        
        # Output options
        st.sidebar.subheader("Output Options")
        show_coordinates = st.sidebar.checkbox("Show Coordinates", True)
        show_descriptions = st.sidebar.checkbox("Show Descriptions", True)
        show_statistics = st.sidebar.checkbox("Show Statistics", True)
        
        # Information about linking
        st.sidebar.subheader("Entity Linking")
        st.sidebar.info("Entities are linked to Wikidata first, then Britannica as fallback. Addresses are linked to OpenStreetMap.")
        
        # Performance options
        st.sidebar.subheader("Performance Options")
        skip_geocoding = st.sidebar.checkbox("Skip geocoding (faster)", False,
                                           help="Skip coordinate lookup to speed up processing")
        limit_entities = st.sidebar.checkbox("Limit to 20 entities", False,
                                           help="Process only first 20 entities for faster results")
        
        return {
            'selected_types': selected_types,
            'show_coordinates': show_coordinates,
            'show_descriptions': show_descriptions,
            'show_statistics': show_statistics,
            'skip_geocoding': skip_geocoding,
            'limit_entities': limit_entities
        }

    def render_input_section(self):
        """Render the text input section."""
        st.header("Input Text")
        
        # Add title input
        analysis_title = st.text_input(
            "Analysis Title (optional)",
            placeholder="Enter a title for this analysis...",
            help="This will be used for naming output files"
        )
        
        # Sample text for demonstration
        sample_text = """Recording the Whitechapel Pavilion in 1961. 191-193 Whitechapel Road. theatre. It was a dauntingly complex task, as to my (then) untrained eye, it appeared to be an impenetrable forest of heavy timbers, movable platforms and hoisting gear, looking like the combined wreckage of half a dozen windmills! Richard Southern's explanations enabled me to allocate names to the various pieces of apparatus. The survey of the Pavilion stage was important at the time because it seemed to be the first time that anything of the kind had been done. Since then, we have learned of complete surviving complexes at, for example, Her Majesty's theatre in London, the Citizens in Glasgow and the Tyne theatre in Newcastle, which has been restored by Dr David Wilmore."""
        
        # Text input area - always shown and editable
        text_input = st.text_area(
            "Enter your text here:",
            value=sample_text,  # Pre-populate with sample text
            height=300,
            placeholder="Paste your text here for entity extraction...",
            help="You can edit this text or replace it with your own content"
        )
        
        # File upload option
        st.subheader("Or upload a text file")
        uploaded_file = st.file_uploader(
            "Choose a text file (optional)",
            type=['txt', 'md'],
            help="Upload a plain text file (.txt) or Markdown file (.md) to replace the text above"
        )
        
        if uploaded_file is not None:
            try:
                uploaded_text = str(uploaded_file.read(), "utf-8")
                text_input = uploaded_text  # Override the text area content
                st.success(f"File uploaded successfully! ({len(uploaded_text)} characters)")
                # Set default title from filename if no title provided
                if not analysis_title:
                    import os
                    default_title = os.path.splitext(uploaded_file.name)[0]
                    st.session_state.suggested_title = default_title
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        # Use suggested title if no title provided
        if not analysis_title and hasattr(st.session_state, 'suggested_title'):
            analysis_title = st.session_state.suggested_title
        elif not analysis_title and not uploaded_file:
            analysis_title = "whitechapel_pavilion_sample"
        
        return text_input, analysis_title or "text_analysis"

    def process_text(self, text: str, title: str, config: Dict[str, Any]):
        """
        Process the input text using the EntityLinker with optimization.
        
        Args:
            text: Input text to process
            config: Configuration dictionary from sidebar
        """
        if not text.strip():
            st.warning("Please enter some text to analyze.")
            return
        
        # Check if we've already processed this exact text
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash == st.session_state.last_processed_hash:
            st.info("This text has already been processed. Results shown below.")
            return
        
        with st.spinner("Processing text and extracting entities..."):
            try:
                # Create a progress bar for the different steps
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Extract entities (cached)
                status_text.text("Extracting entities...")
                progress_bar.progress(25)
                entities = self.cached_extract_entities(text)
                
                # Limit entities if requested for performance
                if config.get('limit_entities', False):
                    entities = entities[:20]
                    st.info("Limited to first 20 entities for faster processing.")
                
                # Step 2: Link to Wikidata (cached)
                status_text.text("Linking to Wikidata...")
                progress_bar.progress(50)
                entities_json = json.dumps(entities, default=str)  # Handle non-serializable objects
                linked_entities_json = self.cached_link_to_wikidata(entities_json)
                entities = json.loads(linked_entities_json)
                
                # Step 3: Link to Britannica (cached)
                status_text.text("Linking to Britannica...")
                progress_bar.progress(75)
                entities_json = json.dumps(entities, default=str)  # Handle non-serializable objects
                linked_entities_json = self.cached_link_to_britannica(entities_json)
                entities = json.loads(linked_entities_json)
                
                # Step 4: Get coordinates (optional for performance)
                if not config.get('skip_geocoding', False):
                    status_text.text("Getting coordinates...")
                    progress_bar.progress(90)
                    # Only geocode first 10 entities to avoid timeout
                    place_entities = [e for e in entities if e['type'] in ['GPE', 'LOCATION', 'FACILITY', 'ORGANIZATION']][:10]
                    for entity in place_entities:
                        if entity in entities:
                            idx = entities.index(entity)
                            try:
                                geocoded = self.entity_linker.get_coordinates([entity])
                                if geocoded and geocoded[0].get('latitude'):
                                    entities[idx] = geocoded[0]
                            except:
                                pass  # Skip if geocoding fails
                
                # Step 5: Generate visualization
                status_text.text("Generating visualization...")
                progress_bar.progress(100)
                html_content = self.create_highlighted_html(text, entities)
                
                # Store in session state
                st.session_state.entities = entities
                st.session_state.processed_text = text
                st.session_state.html_content = html_content
                st.session_state.analysis_title = title
                st.session_state.last_processed_hash = text_hash
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"Processing complete! Found {len(entities)} entities.")
                
                if config.get('skip_geocoding', False):
                    st.info("Geocoding was skipped for faster processing.")
                
            except Exception as e:
                st.error(f"Error processing text: {e}")
                st.exception(e)

    def create_highlighted_html(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """
        Create HTML content with highlighted entities for display.
        
        Args:
            text: Original text
            entities: List of entity dictionaries
            
        Returns:
            HTML string with highlighted entities
        """
        import html as html_module
        
        # Sort entities by start position (reverse for safe replacement)
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        # Start with escaped text
        highlighted = html_module.escape(text)
        
        # Color scheme
        colors = {
            'PERSON': '#FF6B6B',
            'ORGANIZATION': '#4ECDC4', 
            'GPE': '#45B7D1',
            'LOCATION': '#96CEB4',
            'FACILITY': '#FECA57',
            'GSP': '#A55EEA',
            'ADDRESS': '#9B59B6'
        }
        
        # Replace entities from end to start
        for entity in sorted_entities:
            # Only highlight entities that have links
            if not (entity.get('britannica_url') or entity.get('wikidata_url') or entity.get('openstreetmap_url')):
                continue
                
            start = entity['start']
            end = entity['end']
            original_entity_text = text[start:end]
            escaped_entity_text = html_module.escape(original_entity_text)
            color = colors.get(entity['type'], '#CCCCCC')
            
            # Create tooltip with entity information
            tooltip_parts = [f"Type: {entity['type']}"]
            if entity.get('wikidata_description'):
                tooltip_parts.append(f"Description: {entity['wikidata_description']}")
            if entity.get('location_name'):
                tooltip_parts.append(f"Location: {entity['location_name']}")
            
            tooltip = " | ".join(tooltip_parts)
            
            # Create highlighted span with link
            if entity.get('britannica_url'):
                url = html_module.escape(entity["britannica_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('wikidata_url'):
                url = html_module.escape(entity["wikidata_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('openstreetmap_url'):
                url = html_module.escape(entity["openstreetmap_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            
            # Calculate positions in escaped text
            text_before_entity = html_module.escape(text[:start])
            text_entity_escaped = html_module.escape(text[start:end])
            
            escaped_start = len(text_before_entity)
            escaped_end = escaped_start + len(text_entity_escaped)
            
            # Replace in the escaped text
            highlighted = highlighted[:escaped_start] + replacement + highlighted[escaped_end:]
        
        return highlighted

    def render_results(self, config: Dict[str, Any]):
        """
        Render the results section with entities and visualizations.
        
        Args:
            config: Configuration dictionary from sidebar
        """
        if not st.session_state.entities:
            st.info("Enter some text above and click 'Process Text' to see results.")
            return
        
        entities = st.session_state.entities
        filtered_entities = [e for e in entities if e['type'] in config['selected_types']]
        
        st.header("Results")
        
        # Statistics
        if config['show_statistics']:
            self.render_statistics(filtered_entities)
        
        # Highlighted text
        st.subheader("Highlighted Text")
        st.markdown(
            f'<div style="background: white; padding: 20px; border: 1px solid #ddd; border-radius: 5px; line-height: 1.6;">{st.session_state.html_content}</div>',
            unsafe_allow_html=True
        )
        
        # Entity details
        st.subheader("Entity Details")
        self.render_entity_table(filtered_entities, config)
        
        # Map visualization
        if config['show_coordinates']:
            self.render_map(filtered_entities)
        
        # Export options
        self.render_export_section(filtered_entities)

    def render_statistics(self, entities: List[Dict[str, Any]]):
        """Render statistics about the extracted entities."""
        st.subheader("Statistics")
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Entities", len(entities))
        
        with col2:
            linked_count = len([e for e in entities if e.get('wikidata_url') or e.get('britannica_url')])
            st.metric("Linked Entities", linked_count)
        
        with col3:
            geocoded_count = len([e for e in entities if e.get('latitude')])
            st.metric("Geocoded Places", geocoded_count)
        
        with col4:
            unique_types = len(set(e['type'] for e in entities))
            st.metric("Entity Types", unique_types)
        
        # Entity type distribution
        if entities:
            entity_counts = {}
            for entity in entities:
                entity_counts[entity['type']] = entity_counts.get(entity['type'], 0) + 1
            
            # Create pie chart
            fig = px.pie(
                values=list(entity_counts.values()),
                names=list(entity_counts.keys()),
                title="Entity Type Distribution"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

    def render_entity_table(self, entities: List[Dict[str, Any]], config: Dict[str, Any]):
        """Render a table of entity details."""
        if not entities:
            st.info("No entities found matching the selected filters.")
            return
        
        # Prepare data for table
        table_data = []
        for entity in entities:
            row = {
                'Entity': entity['text'],
                'Type': entity['type'],
                'Links': self.format_entity_links(entity)
            }
            
            if config['show_descriptions'] and entity.get('wikidata_description'):
                row['Description'] = entity['wikidata_description']
            
            if config['show_coordinates'] and entity.get('latitude'):
                row['Coordinates'] = f"{entity['latitude']:.4f}, {entity['longitude']:.4f}"
                row['Location'] = entity.get('location_name', '')
            
            table_data.append(row)
        
        # Create DataFrame and display
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

    def format_entity_links(self, entity: Dict[str, Any]) -> str:
        """Format entity links for display in table."""
        links = []
        if entity.get('britannica_url'):
            links.append("Britannica")
        if entity.get('wikidata_url'):
            links.append("Wikidata")
        if entity.get('openstreetmap_url'):
            links.append("OpenStreetMap")
        return " | ".join(links) if links else "No links"

    def render_map(self, entities: List[Dict[str, Any]]):
        """Render a map visualization of geocoded entities."""
        geocoded_entities = [e for e in entities if e.get('latitude')]
        
        if not geocoded_entities:
            st.info("No geocoded entities to display on map.")
            return
        
        st.subheader("Geographic Distribution")
        
        # Prepare data for map
        map_data = []
        for entity in geocoded_entities:
            map_data.append({
                'lat': entity['latitude'],
                'lon': entity['longitude'],
                'name': entity['text'],
                'type': entity['type'],
                'description': entity.get('wikidata_description', ''),
                'location': entity.get('location_name', '')
            })
        
        df_map = pd.DataFrame(map_data)
        
        # Create scatter map
        fig = px.scatter_mapbox(
            df_map,
            lat='lat',
            lon='lon',
            hover_name='name',
            hover_data=['type', 'description'],
            color='type',
            size_max=15,
            zoom=2,
            mapbox_style='open-street-map',
            title='Entity Locations'
        )
        
        fig.update_layout(
            height=500,
            margin={"r":0,"t":30,"l":0,"b":0}
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def render_export_section(self, entities: List[Dict[str, Any]]):
        """Render export options for the results."""
        st.subheader("Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # JSON export
            json_data = {
                "title": st.session_state.analysis_title,
                "entities": entities,
                "metadata": {
                    "total_entities": len(entities),
                    "processed_at": str(pd.Timestamp.now()),
                    "entity_types": list(set(e['type'] for e in entities))
                }
            }
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"{st.session_state.analysis_title}_entities.json",
                mime="application/json"
            )
        
        with col2:
            # CSV export
            if entities:
                df_export = pd.DataFrame([
                    {
                        'entity': e['text'],
                        'type': e['type'],
                        'start': e['start'],
                        'end': e['end'],
                        'wikidata_url': e.get('wikidata_url', ''),
                        'britannica_url': e.get('britannica_url', ''),
                        'description': e.get('wikidata_description', ''),
                        'latitude': e.get('latitude', ''),
                        'longitude': e.get('longitude', ''),
                        'location_name': e.get('location_name', '')
                    }
                    for e in entities
                ])
                
                csv_data = df_export.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"{st.session_state.analysis_title}_entities.csv",
                    mime="text/csv"
                )
        
        with col3:
            # HTML export
            if st.session_state.html_content:
                html_template = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Entity Analysis: {st.session_state.analysis_title}</title>
                    <meta charset="utf-8">
                    <style>
                        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                        .content {{ background: white; padding: 20px; border: 1px solid #ddd; border-radius: 5px; line-height: 1.6; }}
                        .header {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>Entity Analysis: {st.session_state.analysis_title}</h1>
                        <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p>Found {len(entities)} entities</p>
                    </div>
                    <div class="content">
                        {st.session_state.html_content}
                    </div>
                </body>
                </html>
                """
                
                st.download_button(
                    label="Download HTML",
                    data=html_template,
                    file_name=f"{st.session_state.analysis_title}_entities.html",
                    mime="text/html"
                )

    def run(self):
        """Main application runner."""
        # Render header
        self.render_header()
        
        # Render sidebar and get configuration
        config = self.render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Input section
            text_input, analysis_title = self.render_input_section()
            
            # Performance notice
            st.info("💡 Tip: Use performance options in sidebar to speed up processing for large texts.")
            
            # Process button
            if st.button("Process Text", type="primary", use_container_width=True):
                if text_input.strip():
                    self.process_text(text_input, analysis_title, config)
                else:
                    st.warning("Please enter some text to analyze.")
            
            # Quick process button for fast results
            if st.button("Quick Process (No Geocoding)", type="secondary", use_container_width=True):
                if text_input.strip():
                    quick_config = config.copy()
                    quick_config['skip_geocoding'] = True
                    quick_config['limit_entities'] = True
                    self.process_text(text_input, analysis_title, quick_config)
                else:
                    st.warning("Please enter some text to analyze.")
        
        with col2:
            # Results section
            self.render_results(config)


def main():
    """Main function to run the Streamlit application."""
    app = StreamlitEntityLinker()
    app.run()


if __name__ == "__main__":
    main()
