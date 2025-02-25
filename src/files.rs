//! Handles everything related to prompting Gemini with external files.
use reqwest::Method;

use crate::GeminiError;

/// Stores a file used for prompting Gemini
#[derive(Debug, Clone)]
pub struct GeminiFile {
    pub file_uri: String,
    pub mime_type: String,
}
impl GeminiFile {
    pub fn none() -> GeminiFile {
        GeminiFile {
            file_uri: "".to_string(),
            mime_type: "".to_string(),
        }
    }
}

/// Uploads an file to the Google API
///
/// Use a mime filetype from <https://www.iana.org/assignments/media-types/media-types.xhtml>,
/// unfortunately this is neccassary for Google to properly proccess the file.
/// ## Example:
/// ```ignore
/// let api_key = env::var("GEMINI_API_KEY").unwrap();
/// let mut convo = Conversation::new(
///     api_key.clone(),
///     "gemini-1.5-flash".to_string()
/// );
/// let image = upload_image("Testing/cat.png", "image/png", &api_key).await.unwrap();
/// let response = convo.generate_content(vec![
///     Part::Text("Describe this scene".to_string()),
///     Part::File(image)
/// ]).await.unwrap();
/// println!("{0}", response.get_text());
/// ```
pub async fn upload_file<'a>(
    image_path: &'a str,
    mime_type: &'a str,
    api_key: &'a str,
) -> Result<GeminiFile, GeminiError<'a>> {
    let file = std::fs::File::open(image_path)?;
    let file_size = file.metadata().unwrap().len();
    let url = format!(
        "https://generativelanguage.googleapis.com/upload/v1beta/files?key={0}",
        api_key
    );
    let file_name = image_path
        .split("/")
        .last()
        .unwrap()
        .split_once(".")
        .unwrap()
        .0;
    let data = r#"{"file": {"display_name": ""#.to_owned() + file_name + r#""}}"#;

    let client = reqwest::Client::new();

    // Upload metadata to google servers
    let metadata_request = client
        .request(Method::POST, &url)
        .header("X-Goog-Upload-Protocol", "resumable")
        .header("X-Goog-Upload-Command", "start")
        .header("X-Goog-Upload-Header-Content-Length", file_size)
        .header("X-Goog-Upload-Header-Content-Type", mime_type)
        .header("Content-Type", "application/json")
        .body(data)
        .send()
        .await
        .unwrap();

    let metadata_req_headers = metadata_request.headers();
    let upload_url = metadata_req_headers
        .get("x-goog-upload-url")
        .unwrap()
        .to_str()
        .unwrap();

    // Upload the actual bytes
    let _bytes_request = client
        .request(Method::POST, upload_url)
        .header("Content-Length", file_size)
        .header("X-Goog-Upload-Offset", 0)
        .header("X-Goog-Upload-Command", "upload, finalize")
        .body(std::fs::read(image_path).unwrap())
        .send()
        .await;

    // TEST
    let file_list_request = client
        .request(
            Method::GET,
            format!(
                "https://generativelanguage.googleapis.com/v1beta/files/?key={0}",
                std::env::var("GEMINI_API_KEY").unwrap()
            ),
        )
        .send()
        .await
        .unwrap();

    let files_list = &json::parse(&file_list_request.text().await.unwrap()).unwrap()["files"];

    Ok(GeminiFile {
        file_uri: files_list[0]["uri"].as_str().unwrap().to_string(),
        mime_type: mime_type.to_string(),
    })
}
