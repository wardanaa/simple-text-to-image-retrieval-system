# Deployment

Deploy Flask with a production WSGI server such as Gunicorn or uWSGI. Do not use Flask development
mode in production.

Recommended sequence:

1. Create a Python environment and install `requirements.txt`.
2. Configure environment variables and set `FLASK_DEBUG=false`.
3. Build the embedding index before starting the web process.
4. Run the Flask app behind a reverse proxy.
5. Serve static files through the web server where appropriate.
6. Ensure product image files referenced by metadata are readable by the app.
7. Configure model cache storage with enough disk space.
8. Monitor memory use because CLIP model loading is process-local.
9. Use `/health` for basic health checks.

For larger catalogs, consider a separate artifact build step and a vector index in future work.
