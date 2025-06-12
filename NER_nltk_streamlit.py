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

# Configure Streamlit page FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="InsideText: Linking Entities with NLTK",
    layout="centered",  # Changed from "wide" to "centered" for mobile
    initial_sidebar_state="collapsed"  # Changed from "expanded" to "collapsed" for mobile
)

# Authentication is REQUIRED - do not run app without proper login
try:
    import streamlit_authenticator as stauth
    import yaml
    from yaml.loader import SafeLoader
    import os
    
    # Check if config file exists
    if not os.path.exists('config.yaml'):
        st.error("Authentication required: config.yaml file not found!")
        st.info("Please ensure config.yaml is in the same directory as this app.")
        st.stop()
    
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

    # Check if already authenticated via session state
    if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
        name = st.session_state['name']
        authenticator.logout("Logout", "sidebar")
        st.sidebar.success(f"Welcome *{name}*!")
        st.success(f"Successfully logged in as {name}")
        # Continue to app below...
    else:
        # Render login form
        try:
            # Try different login methods
            login_result = None
            try:
                login_result = authenticator.login(location='main')
            except TypeError:
                try:
                    login_result = authenticator.login('Login', 'main')
                except TypeError:
                    login_result = authenticator.login()
            
            # Handle the result
            if login_result is None:
                # Check session state for authentication result
                if 'authentication_status' in st.session_state:
                    auth_status = st.session_state['authentication_status']
                    if auth_status == False:
                        st.error("Username/password is incorrect")
                        st.info("Try username: demo_user with your password")
                    elif auth_status == None:
                        st.warning("Please enter your username and password")
                    elif auth_status == True:
                        st.rerun()  # Refresh to show authenticated state
                else:
                    st.warning("Please enter your username and password")
                st.stop()
            elif isinstance(login_result, tuple) and len(login_result) == 3:
                name, auth_status, username = login_result
                # Store in session state
                st.session_state['authentication_status'] = auth_status
                st.session_state['name'] = name
                st.session_state['username'] = username
                
                if auth_status == True:
                    st.rerun()  # Refresh to show authenticated state
                elif auth_status == False:
                    st.error("Username/password is incorrect")
                    st.stop()
            else:
                st.error(f"Unexpected login result format: {login_result}")
                st.stop()
                
        except Exception as login_error:
            st.error(f"Login method error: {login_error}")
            st.stop()
        
except ImportError:
    st.error("Authentication required: streamlit-authenticator not installed!")
    st.info("Please install streamlit-authenticator to access this application.")
    st.stop()
except Exception as e:
    st.error(f"Authentication error: {e}")
    st.info("Cannot proceed without proper authentication.")
    st.stop()

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

# We'll include the EntityLinker class in this same file instead of importing
# This makes the app self-contained

class EntityLinker:
    """
    Main class for entity linking functionality.
    
    This class handles the complete pipeline from text processing to entity
    extraction, validation, linking, and output generation.
    """
    
    def __init__(self):
        """Initialize the EntityLinker and download required NLTK data."""
        self._download_nltk_data()
        
        # Color scheme for different entity types in HTML output
        self.colors = {
            'PERSON': '#BF7B69',          # F&B Red earth        
            'ORGANIZATION': '#9fd2cd',    # F&B Blue ground
            'GPE': '#C4C3A2',             # F&B Cooking apple green
            'LOCATION': '#EFCA89',        # F&B Yellow ground. 
            'FACILITY': '#C3B5AC',        # F&B Elephants breath
            'GSP': '#C4A998',             # F&B Dead salmon
            'ADDRESS': '#CCBEAA'          # F&B Oxford stone
        }
    
    def _download_nltk_data(self) -> None:
        """Download required NLTK data packages if not already present."""
        import nltk
        import os
        
        # Updated NLTK data requirements for newer versions
        required_downloads = [
            'punkt_tab',  # Updated tokenizer
            'averaged_perceptron_tagger_eng',  # Updated POS tagger
            'maxent_ne_chunker_tab',  # Updated NE chunker
        ]
        
        # Also try old names for compatibility
        compatibility_downloads = [
            'punkt',
            'averaged_perceptron_tagger',
            'maxent_ne_chunker'
        ]
        
        # Words corpus - try to handle corruption
        words_downloads = ['words']
        
        st.info("Downloading NLTK data packages... this may take a moment.")
        
        # Download core packages
        for download_name in required_downloads + compatibility_downloads:
            try:
                nltk.download(download_name, quiet=True)
            except Exception as e:
                # Don't show warnings for compatibility downloads
                if download_name not in compatibility_downloads:
                    st.warning(f"Could not download {download_name}: {e}")
        
        # Handle words corpus separately with error handling
        for download_name in words_downloads:
            try:
                # First try to use existing words corpus
                try:
                    from nltk.corpus import words
                    # Test if it works
                    test_words = words.words()[:10]
                    st.success("Using existing NLTK words corpus")
                    break
                except:
                    pass
                
                # If not available, try to download
                nltk.download(download_name, quiet=True)
                
                # Verify the download worked
                try:
                    from nltk.corpus import words
                    test_words = words.words()[:10]
                    st.success("Successfully downloaded NLTK words corpus")
                except:
                    st.warning("Words corpus download may be corrupted, but entity extraction will still work")
                    
            except Exception as e:
                # If words download fails, continue anyway - it's not critical for NER
                st.warning(f"Could not download words corpus (this is not critical): {e}")
                st.info("Entity extraction will continue without the words corpus")
        
        st.success("NLTK data packages downloaded successfully!")

    def extract_entities(self, text: str):
        """Extract named entities from text using NLTK with proper validation."""
        from nltk import ne_chunk, pos_tag, word_tokenize
        from nltk.tree import Tree
        
        # Step 1: Tokenize and POS tag the text
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        tree = ne_chunk(pos_tags)
        
        entities = []
        word_index = 0
        
        # Step 2: Extract traditional named entities with validation
        for subtree in tree:
            if isinstance(subtree, Tree):
                entity_tokens = [token for token, pos in subtree.leaves()]
                entity_text = ' '.join(entity_tokens)
                entity_label = subtree.label()
                
                # Filter out unwanted entity types
                if entity_label in ['TIME', 'MONEY', 'PERCENT', 'DATE']:
                    word_index += len(entity_tokens)
                    continue
                
                # Validate entity using grammatical context
                if self._is_valid_entity(entity_text, entity_label, pos_tags, word_index, tokens):
                    start_pos = text.find(entity_text)
                    if start_pos != -1:
                        entities.append({
                            'text': entity_text,
                            'type': entity_label,
                            'start': start_pos,
                            'end': start_pos + len(entity_text)
                        })
                
                word_index += len(entity_tokens)
            else:
                word_index += 1
        
        # Step 3: Extract addresses
        addresses = self._extract_addresses(text)
        entities.extend(addresses)
        
        # Step 4: Remove overlapping entities
        entities = self._remove_overlapping_entities(entities)
        
        return entities

    def link_to_britannica(self, entities):
        """Add basic Britannica linking.""" 
        import requests
        import re
        import time
        
        for entity in entities:
            # Skip if already has Wikidata or Wikipedia link
            if entity.get('wikidata_url') or entity.get('wikipedia_url'):
                continue
                
            try:
                search_url = "https://www.britannica.com/search"
                params = {'query': entity['text']}
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(search_url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    # Look for article links
                    pattern = r'href="(/topic/[^"]*)"[^>]*>([^<]*)</a>'
                    matches = re.findall(pattern, response.text)
                    
                    for url_path, link_text in matches:
                        if (entity['text'].lower() in link_text.lower() or 
                            link_text.lower() in entity['text'].lower()):
                            entity['britannica_url'] = f"https://www.britannica.com{url_path}"
                            entity['britannica_title'] = link_text.strip()
                            break
                
                time.sleep(0.3)  # Rate limiting
            except Exception:
                pass
        
        return entities

    def get_coordinates(self, entities):
        """Add coordinate lookup using Python geocoding, then OpenStreetMap as fallback."""
        import requests
        import time
        
        place_types = ['GPE', 'LOCATION', 'FACILITY', 'ORGANIZATION']
        
        for entity in entities:
            if entity['type'] in place_types:
                # Skip if already has coordinates
                if entity.get('latitude') is not None:
                    continue
                    
                # Try Python geocoding libraries first
                if self._try_python_geocoding(entity):
                    continue
                
                # Fall back to OpenStreetMap
                if self._try_openstreetmap(entity):
                    continue
                    
                # If still no coordinates, try a more aggressive search
                self._try_aggressive_geocoding(entity)
        
        return entities
    
    def _try_python_geocoding(self, entity):
        """Try Python geocoding libraries (geopy)."""
        try:
            # Try geopy with multiple providers
            from geopy.geocoders import Nominatim, ArcGIS
            from geopy.exc import GeocoderTimedOut, GeocoderServiceError
            
            # List of geocoders to try in order
            geocoders = [
                ('nominatim', Nominatim(user_agent="EntityLinker/1.0", timeout=10)),
                ('arcgis', ArcGIS(timeout=10)),
            ]
            
            for name, geocoder in geocoders:
                try:
                    # Try the entity name as-is first
                    location = geocoder.geocode(entity['text'], timeout=10)
                    if location:
                        entity['latitude'] = location.latitude
                        entity['longitude'] = location.longitude
                        entity['location_name'] = location.address
                        entity['geocoding_source'] = f'geopy_{name}'
                        return True
                    
                    # If that fails, try with country context for UK places
                    if name == 'nominatim':
                        for suffix in [', UK', ', England', ', Scotland', ', Wales']:
                            location = geocoder.geocode(f"{entity['text']}{suffix}", timeout=10)
                            if location:
                                entity['latitude'] = location.latitude
                                entity['longitude'] = location.longitude
                                entity['location_name'] = location.address
                                entity['geocoding_source'] = f'geopy_{name}_contextual'
                                return True
                        
                    time.sleep(0.3)  # Rate limiting between providers
                except (GeocoderTimedOut, GeocoderServiceError):
                    continue
                except Exception as e:
                    print(f"Geocoding error for {entity['text']} with {name}: {e}")
                    continue
                    
        except ImportError:
            # geopy not installed, skip this method
            pass
        except Exception as e:
            print(f"Python geocoding failed for {entity['text']}: {e}")
            pass
        
        return False
    
    def _try_openstreetmap(self, entity):
        """Fall back to direct OpenStreetMap Nominatim API."""
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': entity['text'],
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            }
            headers = {'User-Agent': 'EntityLinker/1.0'}
        
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    result = data[0]
                    entity['latitude'] = float(result['lat'])
                    entity['longitude'] = float(result['lon'])
                    entity['location_name'] = result['display_name']
                    entity['geocoding_source'] = 'openstreetmap'
                    return True
        
            time.sleep(0.3)  # Rate limiting
        except Exception as e:
            # Debug: print the error for troubleshooting
            print(f"OpenStreetMap geocoding failed for {entity['text']}: {e}")
            pass
        
        return False
    
    def _try_aggressive_geocoding(self, entity):
        """Try more aggressive geocoding with different search terms."""
        import requests
        import time
        
        # Try variations of the entity name
        search_variations = [
            entity['text'],
            f"{entity['text']}, UK",  # Add country for UK places
            f"{entity['text']}, England",
            f"{entity['text']}, Scotland",
            f"{entity['text']}, Wales",
            f"{entity['text']} city",
            f"{entity['text']} town"
        ]
        
        for search_term in search_variations:
            try:
                url = "https://nominatim.openstreetmap.org/search"
                params = {
                    'q': search_term,
                    'format': 'json',
                    'limit': 1,
                    'addressdetails': 1
                }
                headers = {'User-Agent': 'EntityLinker/1.0'}
            
                response = requests.get(url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        result = data[0]
                        entity['latitude'] = float(result['lat'])
                        entity['longitude'] = float(result['lon'])
                        entity['location_name'] = result['display_name']
                        entity['geocoding_source'] = f'openstreetmap_aggressive'
                        entity['search_term_used'] = search_term
                        return True
            
                time.sleep(0.2)  # Rate limiting between attempts
            except Exception:
                continue
        
        return False

    def _is_valid_entity(self, entity_text: str, entity_type: str, 
                       pos_tags, entity_start_word_index: int, tokens) -> bool:
        """Validate an entity by analyzing its grammatical context."""
        # Skip very short entities
        if len(entity_text.strip()) <= 1:
            return False
        
        # Get the POS tag for this entity
        if entity_start_word_index >= len(pos_tags):
            return True  # Default to valid if we can't find the position
        
        word_pos = pos_tags[entity_start_word_index][1]
        entity_word = pos_tags[entity_start_word_index][0]
        
        # Define grammatical categories to filter out
        verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        adjective_tags = ['JJ', 'JJR', 'JJS']
        
        # Filter out words functioning as verbs or adjectives
        if word_pos in verb_tags or word_pos in adjective_tags or word_pos == 'MD':
            return False
        
        # Special handling for sentence-start words (capitalization bias)
        if self._is_sentence_start(tokens, entity_start_word_index):
            return self._validate_sentence_start_entity(
                entity_word, entity_type, pos_tags, 
                entity_start_word_index, verb_tags, adjective_tags
            )
        
        # Additional validation for specific entity types
        if entity_type == 'PERSON':
            return self._validate_person_entity(pos_tags, entity_start_word_index, word_pos)
        
        if entity_type in ['GPE', 'LOCATION', 'FACILITY']:
            return self._validate_place_entity(pos_tags, entity_start_word_index, word_pos)
        
        # Prefer proper nouns as they're more likely to be real entities
        if word_pos in ['NNP', 'NNPS']:
            return True
        
        return True

    def _is_sentence_start(self, tokens, word_index: int) -> bool:
        """Check if a word is at the beginning of a sentence."""
        if word_index == 0:
            return True
        
        # Look back for sentence ending punctuation
        for i in range(word_index - 1, -1, -1):
            token = tokens[i]
            if token in ['.', '!', '?']:
                return True
            elif token.isalpha():  # Found a word before any punctuation
                return False
        
        return True  # If we only found punctuation/symbols, it's sentence start

    def _validate_sentence_start_entity(self, entity_word: str, entity_type: str,
                                      pos_tags, entity_start_word_index: int,
                                      verb_tags, adjective_tags) -> bool:
        """Validate entities that appear at sentence start."""
        try:
            from nltk import pos_tag
            # Check POS tag of lowercase version
            lowercase_tagged = pos_tag([entity_word.lower()])
            natural_pos = lowercase_tagged[0][1]
            
            # Filter out words that would naturally be adjectives/verbs/modals
            if natural_pos in adjective_tags + verb_tags + ['MD']:
                return False
            
            # Check for verb patterns in context
            if entity_start_word_index < len(pos_tags) - 1:
                next_word, next_pos = pos_tags[entity_start_word_index + 1]
                
                # Patterns suggesting verb usage
                if next_pos in ['RB', 'RBR', 'RBS', 'IN', 'TO']:  # followed by adverbs/prepositions
                    return False
                
                # For PERSON: if followed by nouns, could be adjective
                if entity_type == 'PERSON' and next_pos.startswith('NN'):
                    return False
                    
        except Exception:
            pass  # If tagging fails, continue with other checks
        
        return True

    def _validate_person_entity(self, pos_tags, entity_start_word_index: int, word_pos: str) -> bool:
        """Additional validation for PERSON entities."""
        # Check context patterns
        if entity_start_word_index < len(pos_tags) - 1:
            next_word, next_pos = pos_tags[entity_start_word_index + 1]
            # If followed by plural nouns, likely an adjective
            if next_pos in ['NNS', 'NNPS']:
                return False
        
        # Check previous word context
        if entity_start_word_index > 0:
            prev_word, prev_pos = pos_tags[entity_start_word_index - 1]
            
            # If preceded by modal verbs or auxiliaries, likely not a person
            if prev_pos in ['MD', 'VBZ', 'VBP', 'VBD']:
                return False
            
            # If preceded by articles and not proper noun, less likely to be person
            if prev_pos == 'DT' and word_pos not in ['NNP', 'NNPS']:
                return False
        
        return True

    def _validate_place_entity(self, pos_tags, entity_start_word_index: int, word_pos: str) -> bool:
        """Additional validation for place entities."""
        # Check for adjectival usage (e.g., "March weather")
        if entity_start_word_index < len(pos_tags) - 1:
            next_word, next_pos = pos_tags[entity_start_word_index + 1]
            
            # If followed by noun and not proper noun, likely adjectival
            if next_pos.startswith('NN') and word_pos not in ['NNP', 'NNPS']:
                return False
        
        return True

    def _extract_addresses(self, text: str):
        """Extract address patterns that NER might miss."""
        import re
        addresses = []
        
        # Patterns for different address formats
        address_patterns = [
            r'\b\d{1,4}[-–]\d{1,4}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Road|Street|Avenue|Lane|Drive|Way|Place|Square|Gardens)\b',
            r'\b\d{1,4}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Road|Street|Avenue|Lane|Drive|Way|Place|Square|Gardens)\b'
        ]
        
        for pattern in address_patterns:
            for match in re.finditer(pattern, text):
                addresses.append({
                    'text': match.group(),
                    'type': 'ADDRESS',
                    'start': match.start(),
                    'end': match.end()
                })
        
        return addresses

    def _remove_overlapping_entities(self, entities):
        """Remove overlapping entities, keeping the longest ones."""
        entities.sort(key=lambda x: x['start'])
        
        filtered = []
        for entity in entities:
            overlaps = False
            for existing in filtered[:]:  # Create a copy to safely modify during iteration
                # Check if entities overlap
                if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                    # If current entity is longer, remove the existing one
                    if len(entity['text']) > len(existing['text']):
                        filtered.remove(existing)
                        break
                    else:
                        # Current entity is shorter, skip it
                        overlaps = True
                        break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered

    def link_to_wikidata(self, entities):
        """Add basic Wikidata linking."""
        import requests
        import time
        
        for entity in entities:
            try:
                url = "https://www.wikidata.org/w/api.php"
                params = {
                    'action': 'wbsearchentities',
                    'format': 'json',
                    'search': entity['text'],
                    'language': 'en',
                    'limit': 1,
                    'type': 'item'
                }
                
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('search') and len(data['search']) > 0:
                        result = data['search'][0]
                        entity['wikidata_url'] = f"http://www.wikidata.org/entity/{result['id']}"
                        entity['wikidata_description'] = result.get('description', '')
                
                time.sleep(0.1)  # Rate limiting
            except Exception:
                pass  # Continue if API call fails
        
        return entities

    def link_to_wikipedia(self, entities):
        """Add Wikipedia linking for entities without Wikidata links."""
        import requests
        import time
        import urllib.parse
        
        for entity in entities:
            # Skip if already has Wikidata link
            if entity.get('wikidata_url'):
                continue
                
            try:
                # Use Wikipedia's search API
                search_url = "https://en.wikipedia.org/w/api.php"
                search_params = {
                    'action': 'query',
                    'format': 'json',
                    'list': 'search',
                    'srsearch': entity['text'],
                    'srlimit': 1
                }
                
                headers = {'User-Agent': 'EntityLinker/1.0'}
                response = requests.get(search_url, params=search_params, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('query', {}).get('search'):
                        # Get the first search result
                        result = data['query']['search'][0]
                        page_title = result['title']
                        
                        # Create Wikipedia URL
                        encoded_title = urllib.parse.quote(page_title.replace(' ', '_'))
                        entity['wikipedia_url'] = f"https://en.wikipedia.org/wiki/{encoded_title}"
                        entity['wikipedia_title'] = page_title
                        
                        # Get a snippet/description from the search result
                        if result.get('snippet'):
                            # Clean up the snippet (remove HTML tags)
                            import re
                            snippet = re.sub(r'<[^>]+>', '', result['snippet'])
                            entity['wikipedia_description'] = snippet[:200] + "..." if len(snippet) > 200 else snippet
                
                time.sleep(0.2)  # Rate limiting
            except Exception as e:
                print(f"Wikipedia linking failed for {entity['text']}: {e}")
                pass
        
        return entities
        """Add basic Britannica linking.""" 
        import requests
        import re
        import time
        
        for entity in entities:
            # Skip if already has Wikidata link
            if entity.get('wikidata_url'):
                continue
                
            try:
                search_url = "https://www.britannica.com/search"
                params = {'query': entity['text']}
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(search_url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    # Look for article links
                    pattern = r'href="(/topic/[^"]*)"[^>]*>([^<]*)</a>'
                    matches = re.findall(pattern, response.text)
                    
                    for url_path, link_text in matches:
                        if (entity['text'].lower() in link_text.lower() or 
                            link_text.lower() in entity['text'].lower()):
                            entity['britannica_url'] = f"https://www.britannica.com{url_path}"
                            entity['britannica_title'] = link_text.strip()
                            break
                
                time.sleep(0.3)  # Rate limiting
            except Exception:
                pass
        
        return entities

    def link_to_openstreetmap(self, entities):
        """Add OpenStreetMap links to addresses."""
        import requests
        import time
        
        for entity in entities:
            # Only process ADDRESS entities
            if entity['type'] != 'ADDRESS':
                continue
                
            try:
                # Search OpenStreetMap Nominatim for the address
                url = "https://nominatim.openstreetmap.org/search"
                params = {
                    'q': entity['text'],
                    'format': 'json',
                    'limit': 1,
                    'addressdetails': 1
                }
                headers = {'User-Agent': 'EntityLinker/1.0'}
                
                response = requests.get(url, params=params, headers=headers, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        result = data[0]
                        # Create OpenStreetMap link
                        lat = result['lat']
                        lon = result['lon']
                        entity['openstreetmap_url'] = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}&zoom=18"
                        entity['openstreetmap_display_name'] = result['display_name']
                        
                        # Also add coordinates
                        entity['latitude'] = float(lat)
                        entity['longitude'] = float(lon)
                        entity['location_name'] = result['display_name']
                
                time.sleep(0.2)  # Rate limiting
            except Exception:
                pass
        
        return entities


class StreamlitEntityLinker:
    """
    Streamlit wrapper for the EntityLinker class.
    
    Provides a web interface with additional visualization and
    export capabilities for entity analysis.
    """
    
    def __init__(self):
        """Initialize the Streamlit Entity Linker."""
        self.entity_linker = EntityLinker()
        
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
    def cached_link_to_wikipedia(_self, entities_json: str) -> str:
        """Cached Wikipedia linking."""
        import json
        entities = json.loads(entities_json)
        linked_entities = _self.entity_linker.link_to_wikipedia(entities)
        return json.dumps(linked_entities)
    @st.cache_data
    def cached_link_to_britannica(_self, entities_json: str) -> str:
        """Cached Britannica linking."""
        import json
        entities = json.loads(entities_json)
        linked_entities = _self.entity_linker.link_to_britannica(entities)
        return json.dumps(linked_entities)

    def render_header(self):
        """Render the application header."""
        st.title("InsideText: Linking Entities with NLTK")
        st.markdown("""
        **Extract and link named entities from text to external knowledge bases**
        
        This tool uses NLTK for Named Entity Recognition (NER) and links entities to:
        - **Wikidata**: Structured knowledge base
        - **Wikipedia**: Encyclopedia articles (fallback for entities not in Wikidata)
        - **Britannica**: Encyclopedia articles (additional fallback)
        - **OpenStreetMap**: Geographic coordinates and address mapping
        - **Geocoding Services**: Coordinates for places using multiple providers (geopy, Nominatim, ArcGIS)
        """)

    def render_sidebar(self):
        """Render the sidebar with minimal information."""
        # Entity linking information
        st.sidebar.subheader("Entity Linking & Geocoding")
        st.sidebar.info("Entities are linked to Wikidata first, then Wikipedia, then Britannica as fallbacks. Places and addresses are geocoded using multiple services for accurate coordinates.")

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
        sample_text = """ """       
        # Text input area - always shown and editable
        text_input = st.text_area(
            "Enter your text here:",
            value=sample_text,  # Pre-populate with sample text
            height=200,  # Reduced height for mobile
            placeholder="Paste your text here for entity extraction...",
            help="You can edit this text or replace it with your own content"
        )
        
        # File upload option in expander for mobile
        with st.expander("Or upload a text file"):
            uploaded_file = st.file_uploader(
                "Choose a text file",
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
            analysis_title = "text_analysis"
        
        return text_input, analysis_title or "text_analysis"

    def process_text(self, text: str, title: str):
        """
        Process the input text using the EntityLinker.
        
        Args:
            text: Input text to process
            title: Analysis title
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
                
                # Step 2: Link to Wikidata (cached)
                status_text.text("Linking to Wikidata...")
                progress_bar.progress(50)
                entities_json = json.dumps(entities, default=str)  # Handle non-serializable objects
                linked_entities_json = self.cached_link_to_wikidata(entities_json)
                entities = json.loads(linked_entities_json)
                
                # Step 3: Link to Wikipedia (cached)
                status_text.text("Linking to Wikipedia...")
                progress_bar.progress(60)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_wikipedia(entities_json)
                entities = json.loads(linked_entities_json)
                
                # Step 4: Link to Britannica (cached)
                status_text.text("Linking to Britannica...")
                progress_bar.progress(70)
                entities_json = json.dumps(entities, default=str)
                linked_entities_json = self.cached_link_to_britannica(entities_json)
                entities = json.loads(linked_entities_json)
                
                # Step 5: Get coordinates
                status_text.text("Getting coordinates...")
                progress_bar.progress(85)
                # Geocode all place entities more aggressively
                place_entities = [e for e in entities if e['type'] in ['GPE', 'LOCATION', 'FACILITY', 'ORGANIZATION']]
                
                if place_entities:
                    try:
                        # Use the get_coordinates method which handles multiple geocoding services
                        geocoded_entities = self.entity_linker.get_coordinates(place_entities)
                        
                        # Update the entities list with geocoded results
                        for geocoded_entity in geocoded_entities:
                            # Find the corresponding entity in the main list and update it
                            for idx, entity in enumerate(entities):
                                if (entity['text'] == geocoded_entity['text'] and 
                                    entity['type'] == geocoded_entity['type'] and
                                    entity['start'] == geocoded_entity['start']):
                                    entities[idx] = geocoded_entity
                                    break
                    except Exception as e:
                        st.warning(f"Some geocoding failed: {e}")
                        # Continue with processing even if geocoding fails
                
                # Step 6: Link addresses to OpenStreetMap
                status_text.text("Linking addresses to OpenStreetMap...")
                progress_bar.progress(90)
                entities = self.entity_linker.link_to_openstreetmap(entities)
                
                # Step 7: Generate visualization
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
            'PERSON': '#BF7B69',          # F&B Red earth        
            'ORGANIZATION': '#9fd2cd',    # F&B Blue ground
            'GPE': '#C4C3A2',             # F&B Cooking apple green
            'LOCATION': '#EFCA89',        # F&B Yellow ground 
            'FACILITY': '#C3B5AC',        # F&B Elephants breath
            'GSP': '#C4A998',             # F&B Dead salmon
            'ADDRESS': '#CCBEAA'          # F&B Oxford stone
        }
        
        # Replace entities from end to start
        for entity in sorted_entities:
            # Highlight entities that have links OR coordinates
            has_links = (entity.get('britannica_url') or 
                        entity.get('wikidata_url') or 
                        entity.get('openstreetmap_url'))
            has_coordinates = entity.get('latitude') is not None
            
            if not (has_links or has_coordinates):
                continue
                
            start = entity['start']
            end = entity['end']
            original_entity_text = text[start:end]
            escaped_entity_text = html_module.escape(original_entity_text)
            color = colors.get(entity['type'], '#E7E2D2')
            
            # Create tooltip with entity information
            tooltip_parts = [f"Type: {entity['type']}"]
            if entity.get('wikidata_description'):
                tooltip_parts.append(f"Description: {entity['wikidata_description']}")
            if entity.get('location_name'):
                tooltip_parts.append(f"Location: {entity['location_name']}")
            
            tooltip = " | ".join(tooltip_parts)
            
            # Create highlighted span with link (priority: Wikipedia > Wikidata > Britannica > OpenStreetMap > Coordinates only)
            if entity.get('wikipedia_url'):
                url = html_module.escape(entity["wikipedia_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('wikidata_url'):
                url = html_module.escape(entity["wikidata_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            elif entity.get('britannica_url'):
                url = html_module.escape(entity["openstreetmap_url"])
                replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
            else:
                # Just highlight with coordinates (no link)
                replacement = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;" title="{tooltip}">{escaped_entity_text}</span>'
            
            # Calculate positions in escaped text
            text_before_entity = html_module.escape(text[:start])
            text_entity_escaped = html_module.escape(text[start:end])
            
            escaped_start = len(text_before_entity)
            escaped_end = escaped_start + len(text_entity_escaped)
            
            # Replace in the escaped text
            highlighted = highlighted[:escaped_start] + replacement + highlighted[escaped_end:]
        
        return highlighted

    def render_results(self):
        """Render the results section with entities and visualizations."""
        if not st.session_state.entities:
            st.info("Enter some text above and click 'Process Text' to see results.")
            return
        
        entities = st.session_state.entities
        
        st.header("Results")
        
        # Highlighted text
        st.subheader("Highlighted Text")
        if st.session_state.html_content:
            st.markdown(
                st.session_state.html_content,
                unsafe_allow_html=True
            )
        else:
            st.info("No highlighted text available. Process some text first.")
        
        # Entity details in collapsible section for mobile
        with st.expander("Entity Details", expanded=False):
            self.render_entity_table(entities)
        
        # Export options in collapsible section for mobile
        with st.expander("Export Results", expanded=False):
            self.render_export_section(entities)

    def render_statistics(self, entities: List[Dict[str, Any]]):
        """Render statistics about the extracted entities."""
        # Create columns for metrics (works well on mobile)
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Entities", len(entities))
            geocoded_count = len([e for e in entities if e.get('latitude')])
            st.metric("Geocoded Places", geocoded_count)
        
        with col2:
            linked_count = len([e for e in entities if e.get('wikidata_url') or e.get('wikipedia_url') or e.get('britannica_url')])
            st.metric("Linked Entities", linked_count)
            unique_types = len(set(e['type'] for e in entities))
            st.metric("Entity Types", unique_types)

    def render_entity_table(self, entities: List[Dict[str, Any]]):
        """Render a table of entity details."""
        if not entities:
            st.info("No entities found.")
            return
        
        # Prepare data for table
        table_data = []
        for entity in entities:
            row = {
                'Entity': entity['text'],
                'Type': entity['type'],
                'Links': self.format_entity_links(entity)
            }
            
            if entity.get('wikidata_description'):
                row['Description'] = entity['wikidata_description']
            elif entity.get('wikipedia_description'):
                row['Description'] = entity['wikipedia_description']
            elif entity.get('britannica_title'):
                row['Description'] = entity['britannica_title']
            
            if entity.get('latitude'):
                row['Coordinates'] = f"{entity['latitude']:.4f}, {entity['longitude']:.4f}"
                row['Location'] = entity.get('location_name', '')
            
            table_data.append(row)
        
        # Create DataFrame and display
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

    def format_entity_links(self, entity: Dict[str, Any]) -> str:
        """Format entity links for display in table."""
        links = []
        if entity.get('wikipedia_url'):
            links.append("Wikipedia")
        if entity.get('wikidata_url'):
            links.append("Wikidata")
        if entity.get('britannica_url'):
            links.append("Britannica")
        if entity.get('openstreetmap_url'):
            links.append("OpenStreetMap")
        return " | ".join(links) if links else "No links"

    def render_export_section(self, entities: List[Dict[str, Any]]):
        """Render export options for the results."""
        # Stack buttons vertically for mobile
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export - create JSON-LD format
            json_data = {
                "@context": "http://schema.org/",
                "@type": "TextDigitalDocument",
                "text": st.session_state.processed_text,
                "dateCreated": str(pd.Timestamp.now().isoformat()),
                "title": st.session_state.analysis_title,
                "entities": []
            }
            
            # Format entities for JSON-LD
            for entity in entities:
                entity_data = {
                    "name": entity['text'],
                    "type": entity['type'],
                    "startOffset": entity['start'],
                    "endOffset": entity['end']
                }
                
                if entity.get('wikidata_url'):
                    entity_data['sameAs'] = entity['wikidata_url']
                
                if entity.get('wikidata_description'):
                    entity_data['description'] = entity['wikidata_description']
                elif entity.get('wikipedia_description'):
                    entity_data['description'] = entity['wikipedia_description']
                elif entity.get('britannica_title'):
                    entity_data['description'] = entity['britannica_title']
                
                if entity.get('latitude') and entity.get('longitude'):
                    entity_data['geo'] = {
                        "@type": "GeoCoordinates",
                        "latitude": entity['latitude'],
                        "longitude": entity['longitude']
                    }
                    if entity.get('location_name'):
                        entity_data['geo']['name'] = entity['location_name']
                
                if entity.get('wikipedia_url'):
                    if 'sameAs' in entity_data:
                        if isinstance(entity_data['sameAs'], str):
                            entity_data['sameAs'] = [entity_data['sameAs'], entity['wikipedia_url']]
                        else:
                            entity_data['sameAs'].append(entity['wikipedia_url'])
                    else:
                        entity_data['sameAs'] = entity['wikipedia_url']
                
                if entity.get('britannica_url'):
                    if 'sameAs' in entity_data:
                        if isinstance(entity_data['sameAs'], str):
                            entity_data['sameAs'] = [entity_data['sameAs'], entity['britannica_url']]
                        else:
                            entity_data['sameAs'].append(entity['britannica_url'])
                    else:
                        entity_data['sameAs'] = entity['britannica_url']
                
                if entity.get('openstreetmap_url'):
                    if 'sameAs' in entity_data:
                        if isinstance(entity_data['sameAs'], str):
                            entity_data['sameAs'] = [entity_data['sameAs'], entity['openstreetmap_url']]
                        else:
                            entity_data['sameAs'].append(entity['openstreetmap_url'])
                    else:
                        entity_data['sameAs'] = entity['openstreetmap_url']
                
                json_data['entities'].append(entity_data)
            
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="Download JSON-LD",
                data=json_str,
                file_name=f"{st.session_state.analysis_title}_entities.jsonld",
                mime="application/ld+json",
                use_container_width=True
            )
        
        with col2:
            # HTML export
            if st.session_state.html_content:
                html_template = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Entity Analysis: {st.session_state.analysis_title}</title>
                    <meta charset="utf-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <style>
                        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                        .content {{ background: white; padding: 20px; border: 1px solid #ddd; border-radius: 5px; line-height: 1.6; }}
                        .header {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                        @media (max-width: 768px) {{
                            body {{ padding: 10px; }}
                            .content {{ padding: 15px; }}
                            .header {{ padding: 10px; }}
                        }}
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
                    mime="text/html",
                    use_container_width=True
                )

    def run(self):
        """Main application runner."""
        # Render header
        self.render_header()
        
        # Render sidebar
        self.render_sidebar()
        
        # Single column layout for mobile compatibility
        # Input section
        text_input, analysis_title = self.render_input_section()
        
        # Process button
        if st.button("Process Text", type="primary", use_container_width=True):
            if text_input.strip():
                self.process_text(text_input, analysis_title)
            else:
                st.warning("Please enter some text to analyze.")
        
        # Add some spacing
        st.markdown("---")
        
        # Results section
        self.render_results()


def main():
    """Main function to run the Streamlit application."""
    app = StreamlitEntityLinker()
    app.run()


if __name__ == "__main__":
    main()
