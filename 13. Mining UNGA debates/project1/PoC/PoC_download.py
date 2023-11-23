from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

import requests
import time
import os


class StatementDownloader:
    def __init__(
        self,
        url: str = "https://gadebate.un.org/en/sessions-archive?f%5B0%5D=choose_a_session_ungad%3A564",
        download_dir: str = "corpora/pdfs",
    ) -> None:
        self.url = url
        self.download_dir = download_dir
        self.attributes = {}
        os.makedirs(self.download_dir, exist_ok=True)
        self.driver = self._create_driver()

    def _create_driver(self) -> webdriver.Chrome:
        options = Options()
        options.add_argument("--start-maximized")
        options.add_argument("--window-size=1920,1080")
        driver = webdriver.Chrome(options=options)
        return driver

    def download(self) -> None:
        self.driver.get(self.url)

        # find all country links
        links = self.driver.find_elements(By.TAG_NAME, "a")

        # filter only those that start with /en/78
        country_links = [
            link for link in links if link.get_attribute("href").find("/en/78") != -1
        ]

        # visit all country links
        for country_link in country_links:
            # go to the country link
            self.driver.get(country_link.get_attribute("href"))

            # find the statement link
            statement_links = self.driver.find_elements(By.TAG_NAME, "a")

            english_statement = [
                statement
                for statement in statement_links
                if statement.get_attribute("href") is not None
                and statement.get_attribute("href").find("en.pdf") != -1
            ]

            if len(english_statement) == 0:
                print(
                    f"No english statement found for this country. ({self.driver.current_url})"
                )
            else:
                pdf_link = english_statement[0].get_attribute("href")
                print(f"Found english statement this country at {pdf_link}")

                # Get additional attributes
                pdf_name = pdf_link.split("/")[-1]

                speaker_name = self.driver.find_element(
                    By.CLASS_NAME, "field--name-field-speaker-name"
                ).text
                speaker_position_div = self.driver.find_element(
                    By.CLASS_NAME, "field--name-field-speaker-function-2"
                )
                speaker_position = speaker_position_div.find_element(
                    By.TAG_NAME, "div"
                ).text
                self.attributes[pdf_name] = [speaker_name, speaker_position]
                print(f"Speaker: {speaker_name}")
                print(f"Position: {speaker_position}")

            # go back and save the pdf
            self.driver.back()
            pdf_path = os.path.join(self.download_dir, pdf_name)
            with open(pdf_path, "wb") as f:
                f.write(requests.get(pdf_link).content)
        self.driver.close()
        # save the attributes
        with open(os.path.join(self.download_dir, "attributes.csv"), "w") as f:
            f.write("pdf_name,speaker_name,speaker_position\n")
            for pdf_name, attributes in self.attributes.items():
                f.write(f"{pdf_name},{attributes[0]},{attributes[1]}\n")


def main():
    downloader = StatementDownloader()
    downloader.download()


if __name__ == "__main__":
    main()
