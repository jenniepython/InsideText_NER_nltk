def extract_entities(self, text: str):
    """Extract named entities from text using NLTK with proper validation."""
    from nltk import ne_chunk, pos_tag, word_tokenize
    from nltk.tree import Tree
    import re
    
    # Step 1: Tokenize and POS tag the text
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    tree = ne_chunk(pos_tags)
    
    # Step 2: First pass - extract unique entities from NLTK
    unique_entities = {}  # Dictionary to store unique entity texts and their properties
    word_index = 0
    
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
                # Store unique entity information
                entity_key = f"{entity_text.lower()}:{entity_label}"
                if entity_key not in unique_entities:
                    unique_entities[entity_key] = {
                        'text': entity_text,
                        'type': entity_label,
                        'original_case': entity_text  # Keep original case
                    }
            
            word_index += len(entity_tokens)
        else:
            word_index += 1
    
    # Step 3: Find ALL occurrences of each unique entity in the text
    entities = []
    
    for entity_key, entity_info in unique_entities.items():
        entity_text = entity_info['text']
        entity_type = entity_info['type']
        
        # Find all occurrences of this entity in the text (case-insensitive search but preserve original case)
        # Use regex to find word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(entity_text) + r'\b'
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Use the actual text from the match (preserves case as it appears in the source)
            actual_text = text[match.start():match.end()]
            
            entities.append({
                'text': actual_text,
                'type': entity_type,
                'start': match.start(),
                'end': match.end()
            })
    
    # Step 4: Extract addresses
    addresses = self._extract_addresses(text)
    entities.extend(addresses)
    
    # Step 5: Remove overlapping entities (but keep multiple occurrences of same entity)
    entities = self._remove_overlapping_entities(entities)
    
    # Step 6: Sort by position for consistent processing
    entities.sort(key=lambda x: x['start'])
    
    return entities

def create_highlighted_html(self, text: str, entities: List[Dict[str, Any]]) -> str:
    """
    Create HTML content with highlighted entities for display.
    This version properly handles multiple occurrences of the same entity.
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
    
    # Create a mapping of entity text to link information for consistent linking
    entity_link_map = {}
    
    # First, collect link information for each unique entity text + type combination
    for entity in entities:
        entity_key = f"{entity['text'].lower()}:{entity['type']}"
        if entity_key not in entity_link_map:
            # Store the link information from the first occurrence that has links
            has_links = (entity.get('britannica_url') or 
                         entity.get('wikidata_url') or 
                         entity.get('wikipedia_url') or     
                         entity.get('openstreetmap_url'))
            has_coordinates = entity.get('latitude') is not None
            
            if has_links or has_coordinates:
                entity_link_map[entity_key] = entity
    
    # Now process entities from end to start to avoid position shifting
    for entity in sorted_entities:
        entity_key = f"{entity['text'].lower()}:{entity['type']}"
        
        # Use the link information from our map
        if entity_key not in entity_link_map:
            continue
            
        link_entity = entity_link_map[entity_key]
        
        start = entity['start']
        end = entity['end']
        original_entity_text = text[start:end]
        escaped_entity_text = html_module.escape(original_entity_text)
        color = colors.get(entity['type'], '#E7E2D2')
        
        # Create tooltip with entity information
        tooltip_parts = [f"Type: {entity['type']}"]
        if link_entity.get('wikidata_description'):
            tooltip_parts.append(f"Description: {link_entity['wikidata_description']}")
        elif link_entity.get('wikipedia_description'):
            tooltip_parts.append(f"Description: {link_entity['wikipedia_description']}")
        elif link_entity.get('britannica_title'):
            tooltip_parts.append(f"Description: {link_entity['britannica_title']}")
        
        if link_entity.get('location_name'):
            tooltip_parts.append(f"Location: {link_entity['location_name']}")
        
        tooltip = " | ".join(tooltip_parts)
        
        # Create highlighted span with link (priority: Wikipedia > Wikidata > Britannica > OpenStreetMap > Coordinates only)
        if link_entity.get('wikipedia_url'):
            url = html_module.escape(link_entity["wikipedia_url"])
            replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
        elif link_entity.get('wikidata_url'):
            url = html_module.escape(link_entity["wikidata_url"])
            replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
        elif link_entity.get('britannica_url'):
            url = html_module.escape(link_entity["britannica_url"])
            replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
        elif link_entity.get('openstreetmap_url'):
            url = html_module.escape(link_entity["openstreetmap_url"])
            replacement = f'<a href="{url}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; text-decoration: none; color: black;" target="_blank" title="{tooltip}">{escaped_entity_text}</a>'
        else:
            # Just highlight with coordinates (no link)
            replacement = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;" title="{tooltip}">{escaped_entity_text}</span>'
        
        # Calculate positions in escaped text
        # We need to be more careful about position calculation in escaped text
        text_before_entity = text[:start]
        escaped_text_before = html_module.escape(text_before_entity)
        text_entity = text[start:end]
        escaped_text_entity = html_module.escape(text_entity)
        
        escaped_start = len(escaped_text_before)
        escaped_end = escaped_start + len(escaped_text_entity)
        
        # Replace in the escaped text
        highlighted = highlighted[:escaped_start] + replacement + highlighted[escaped_end:]
    
    return highlighted

def link_to_wikidata(self, entities):
    """Add basic Wikidata linking - now handles multiple occurrences efficiently."""
    import requests
    import time
    
    # Create a dictionary to store link information by entity text + type
    entity_links = {}
    
    # Get unique entity text + type combinations to avoid duplicate API calls
    unique_entities = {}
    for entity in entities:
        entity_key = f"{entity['text'].lower()}:{entity['type']}"
        if entity_key not in unique_entities:
            unique_entities[entity_key] = entity
    
    # Process unique entities
    for entity_key, entity in unique_entities.items():
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
                    entity_links[entity_key] = {
                        'wikidata_url': f"http://www.wikidata.org/entity/{result['id']}",
                        'wikidata_description': result.get('description', '')
                    }
            
            time.sleep(0.1)  # Rate limiting
        except Exception:
            pass  # Continue if API call fails
    
    # Apply the link information to all entities
    for entity in entities:
        entity_key = f"{entity['text'].lower()}:{entity['type']}"
        if entity_key in entity_links:
            entity.update(entity_links[entity_key])
    
    return entities

def link_to_wikipedia(self, entities):
    """Add Wikipedia linking for entities without Wikidata links - optimized for multiple occurrences."""
    import requests
    import time
    import urllib.parse
    
    # Create a dictionary to store link information by entity text + type
    entity_links = {}
    
    # Get unique entity text + type combinations to avoid duplicate API calls
    unique_entities = {}
    for entity in entities:
        entity_key = f"{entity['text'].lower()}:{entity['type']}"
        if entity_key not in unique_entities and not entity.get('wikidata_url'):
            unique_entities[entity_key] = entity
    
    # Process unique entities that don't have Wikidata links
    for entity_key, entity in unique_entities.items():
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
                    
                    entity_links[entity_key] = {
                        'wikipedia_url': f"https://en.wikipedia.org/wiki/{encoded_title}",
                        'wikipedia_title': page_title
                    }
                    
                    # Get a snippet/description from the search result
                    if result.get('snippet'):
                        # Clean up the snippet (remove HTML tags)
                        import re
                        snippet = re.sub(r'<[^>]+>', '', result['snippet'])
                        entity_links[entity_key]['wikipedia_description'] = snippet[:200] + "..." if len(snippet) > 200 else snippet
            
            time.sleep(0.2)  # Rate limiting
        except Exception as e:
            print(f"Wikipedia linking failed for {entity['text']}: {e}")
            pass
    
    # Apply the link information to all entities
    for entity in entities:
        entity_key = f"{entity['text'].lower()}:{entity['type']}"
        if entity_key in entity_links:
            entity.update(entity_links[entity_key])
    
    return entities

def link_to_britannica(self, entities):
    """Add basic Britannica linking - optimized for multiple occurrences.""" 
    import requests
    import re
    import time
    
    # Create a dictionary to store link information by entity text + type
    entity_links = {}
    
    # Get unique entity text + type combinations to avoid duplicate API calls
    unique_entities = {}
    for entity in entities:
        entity_key = f"{entity['text'].lower()}:{entity['type']}"
        if (entity_key not in unique_entities and 
            not entity.get('wikidata_url') and 
            not entity.get('wikipedia_url')):
            unique_entities[entity_key] = entity
    
    # Process unique entities that don't have other links
    for entity_key, entity in unique_entities.items():            
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
                        entity_links[entity_key] = {
                            'britannica_url': f"https://www.britannica.com{url_path}",
                            'britannica_title': link_text.strip()
                        }
                        break
            
            time.sleep(0.3)  # Rate limiting
        except Exception:
            pass
    
    # Apply the link information to all entities
    for entity in entities:
        entity_key = f"{entity['text'].lower()}:{entity['type']}"
        if entity_key in entity_links:
            entity.update(entity_links[entity_key])
    
    return entities
