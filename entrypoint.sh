#!/bin/sh
set -e

FLAG=/home/node/.n8n/.imported

if [ ! -f "$FLAG" ]; then
  echo "==> First boot: importing credentials & workflow"

  cat > /tmp/credentials.json <<EOF
[
  {
    "id": "3J7FRzJpFuxgYqqm",
    "name": "MySQL account",
    "type": "mySql",
    "data": {
      "host": "mysql",
      "database": "$MYSQL_DATABASE",
      "user": "$MYSQL_USER",
      "password": "$MYSQL_PASSWORD",
      "port": 3306,
      "ssl": false
    }
  },
  {
    "id": "aPFwuENmDaq6206Q",
    "name": "Groq account",
    "type": "groqApi",
    "data": { "apiKey": "$GROQ_API_KEY" }
  }
]
EOF

  n8n import:credentials --input=/tmp/credentials.json
  n8n import:workflow --input=/import/workflow.json

  echo "==> Activating workflow"
  n8n update:workflow --id=h2fHjNoYWb25xhsp --active=true \
    || n8n publish:workflow --id=h2fHjNoYWb25xhsp \
    || echo "!! Could not auto-activate — activate manually once in the n8n UI, then it persists across restarts."

  rm -f /tmp/credentials.json
  touch "$FLAG"
  echo "==> Import complete"
else
  echo "==> Already imported, skipping"
fi

exec n8n start
