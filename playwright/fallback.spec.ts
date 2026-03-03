import { test, expect } from "@playwright/test";

test("landing loads", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "accel-gpu", level: 1 })).toBeVisible();
});

test("demo reports backend", async ({ page }) => {
  await page.goto("/example/");
  await expect(page.getByText(/Backend:\s*(webgpu|webgl|cpu)/i)).toBeVisible({ timeout: 30_000 });
});

test("playground run button executes", async ({ page }) => {
  await page.goto("/playground/");
  await page.getByRole("button", { name: "Run" }).click();
  await expect(page.locator("#output")).not.toHaveText("Click Run to execute.", {
    timeout: 30_000,
  });
});
