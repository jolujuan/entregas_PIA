#!/bin/bash

# Script de prueba para la API de Sentimientos con Flask
# Asegúrate de que Flask está corriendo en http://localhost:5000

API_URL="http://localhost:5000"

echo "============================================"
echo "Pruebas de la API de Sentimientos"
echo "============================================"
echo ""

# Test 1: Verificar que la API está viva
echo "TEST 1: Verificar que la API está viva"
echo "---"
curl -s "$API_URL/" | python -m json.tool
echo ""
echo ""

# Test 2: Predicción con un único texto (Positivo)
echo "TEST 2: Predicción con texto positivo"
echo "---"
curl -s -X POST "$API_URL/predict_sentiment" \
  -H "Content-Type: application/json" \
  -d '{"text": "What a great product! Love it!"}' | python -m json.tool
echo ""
echo ""

# Test 3: Predicción con un único texto (Negativo)
echo "TEST 3: Predicción con texto negativo"
echo "---"
curl -s -X POST "$API_URL/predict_sentiment" \
  -H "Content-Type: application/json" \
  -d '{"text": "Terrible service, horrible day... #fail"}' | python -m json.tool
echo ""
echo ""

# Test 4: Predicción con un único texto (Neutral/Ambiguo)
echo "TEST 4: Predicción con texto neutral"
echo "---"
curl -s -X POST "$API_URL/predict_sentiment" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is just OK, not great, not bad."}' | python -m json.tool
echo ""
echo ""

# Test 5: Predicción con múltiples textos
echo "TEST 5: Predicción con múltiples textos (lista)"
echo "---"
curl -s -X POST "$API_URL/predict_sentiment" \
  -H "Content-Type: application/json" \
  -d '{"text": ["What a great product! Love it!", "Terrible service, horrible day... #fail", "This is just OK, not great, not bad."]}' | python -m json.tool
echo ""
echo ""

# Test 6: Error - Falta la clave 'text'
echo "TEST 6: Error - Falta la clave 'text'"
echo "---"
curl -s -X POST "$API_URL/predict_sentiment" \
  -H "Content-Type: application/json" \
  -d '{"mensaje": "Sin la clave text"}' | python -m json.tool
echo ""
echo ""

# Test 7: Error - Request no es JSON
echo "TEST 7: Error - Request no es JSON"
echo "---"
curl -s -X POST "$API_URL/predict_sentiment" \
  -H "Content-Type: text/plain" \
  -d 'Este es un texto plano' | python -m json.tool
echo ""
echo "============================================"
echo "Pruebas completadas"
echo "============================================"